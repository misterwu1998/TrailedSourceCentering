import numpy as np
import torch
import matplotlib.pyplot as plt
from photutils import psf
from astropy import stats
import trailed_source_centering

Q = 59
N = 256
PRF_OVERSAMPLE = 20
PRF_RADIUS = 20

FLOAT_FMT = '%16.7f'

def save_control_points(t_seq, control_points_x, control_points_y, csvpath):
    arr = np.stack((t_seq,control_points_x,control_points_y), axis=0)
    np.savetxt(csvpath,arr,FLOAT_FMT,delimiter=',')
def load_control_points(csvpath):
    arr = np.loadtxt(csvpath,dtype=np.float64,delimiter=',')
    return arr[0,:], arr[1,:], arr[2,:]

def broadcast(a:np.ndarray, b:np.ndarray):
    a=np.expand_dims(np.array(a),axis=0)
    b=np.expand_dims(np.array(b),axis=0)
    a_shape=list(np.shape(a))
    b_shape=list(np.shape(b))
    broadcast_shape=a_shape+b_shape[1:]
    a_new_shape=np.copy(broadcast_shape); a_new_shape[len(a_shape):]=1
    b_new_shape=np.copy(broadcast_shape); b_new_shape[:len(a_shape)]=1
    a=np.reshape(a,a_new_shape)
    b=np.reshape(b,b_new_shape)
    a,b=np.broadcast_arrays(a,b)
    a=a[0,...]
    b=b[0,...]
    return (a,b)

def batch_convert_from_ndarray(*arrays):
    tensors=[]
    for arr in arrays:
        tensors.append(torch.from_numpy(arr))
    return tuple(tensors)

def batch_convert_to_ndarray(*tensors):
    arrays=[]
    for t in tensors:
        arrays.append(t.numpy())
    return tuple(arrays)

# Generate a Gaussian PRF model
prf_arr_size = 2*PRF_RADIUS*PRF_OVERSAMPLE + 1
ygrid, xgrid = np.mgrid[:prf_arr_size, :prf_arr_size]
xgrid = xgrid/PRF_OVERSAMPLE; ygrid = ygrid/PRF_OVERSAMPLE
xgrid -= PRF_RADIUS; ygrid -= PRF_RADIUS
gaussian_prf = psf.GaussianPRF(flux=1.0, x_0=0.0, y_0=0.0, x_fwhm=2.0, y_fwhm=2.0)
gaussian_prf_data = gaussian_prf(xgrid, ygrid)

# Generate a trajectory
control_points_t = np.linspace(-50.0, 50.0, Q, endpoint=True)
control_points_x = control_points_t+50
control_points_y = (control_points_t+50)**2 / 100
control_points_x += 10; control_points_y += 10
tt = np.linspace(-50.0, 50.0, 2*N+1, endpoint=True)
xx = np.interp(tt, control_points_t, control_points_x)
yy = np.interp(tt, control_points_t, control_points_y)

# Synthesize an image
f_star = 100.0
ygrid, xgrid = np.mgrid[:120, :120]
xgrid_, xx_ = broadcast(xgrid, xx)
ygrid_, yy_ = broadcast(ygrid, yy)
prf_value = gaussian_prf(xgrid_-xx_, ygrid_-yy_)
img = f_star * np.sum(prf_value, axis=-1)
img += 100.0 # background
img += (np.random.randn(120,120) * f_star/10.0) # noise

# ROI
sigma_clipped_img_data = stats.sigma_clip(img, sigma_upper=3.0, maxiters=3)
bg_mean = sigma_clipped_img_data.mean(); print('bg_mean: ', bg_mean)
bg_std = sigma_clipped_img_data.std()
fg = img-bg_mean
roi_mask = (img >= bg_mean + 2.5*bg_std)
# Discard false detections... (omitted)
pixels_x = (xgrid[roi_mask]).astype(np.float64)
pixels_y = (ygrid[roi_mask]).astype(np.float64)
pixels_fg = (fg[roi_mask]).astype(np.float64)

starting_control_points_t = np.linspace(-50.0, 50.0, 3, endpoint=True, dtype=np.float64)
starting_control_points_x = np.interp(starting_control_points_t, control_points_t, control_points_x).astype(np.float64)
starting_control_points_y = np.interp(starting_control_points_t, control_points_t, control_points_y).astype(np.float64)
starting_control_points_x += (np.random.rand(3)*2 - 1.0)
starting_control_points_y += (np.random.rand(3)*2 - 1.0)

prf_functor = trailed_source_centering.wrap_psf_functor(torch.from_numpy(gaussian_prf_data), (PRF_RADIUS*PRF_OVERSAMPLE, PRF_RADIUS*PRF_OVERSAMPLE), PRF_OVERSAMPLE)
binning_grid_x = np.array([[0.0]], dtype=np.float64); binning_grid_y = binning_grid_x
def _loop_exit(optimized_control_points_x:torch.Tensor, optimized_control_points_y:torch.Tensor, t_seq:torch.Tensor):
    grad_x = optimized_control_points_x[1:]-optimized_control_points_x[:-1]
    grad_y = optimized_control_points_y[1:]-optimized_control_points_y[:-1]
    stride = torch.sqrt(grad_x**2 + grad_y**2)
    mean_stride = torch.mean(stride)
    result = mean_stride<=2.0
    print('Control points number: {}; Exit: {}'.format(t_seq.size(dim=0),result))
    return result

starting_control_points_x,starting_control_points_y,starting_control_points_t,pixels_x,pixels_y,pixels_fg,binning_grid_x,binning_grid_y=batch_convert_from_ndarray(
    starting_control_points_x,starting_control_points_y,starting_control_points_t,pixels_x,pixels_y,pixels_fg,binning_grid_x,binning_grid_y
)
optimized_control_points_x,optimized_control_points_y,refined_t_seq = trailed_source_centering.main_loop(
        starting_control_points_x,starting_control_points_y,starting_control_points_t,
        pixels_x,pixels_y,pixels_fg,
        f_star,prf_functor,
        0.09, 0.01,
        512,
        binning_grid_x, binning_grid_y,
        _loop_exit
)
optimized_control_points_x,optimized_control_points_y,refined_t_seq = batch_convert_to_ndarray(
    optimized_control_points_x,optimized_control_points_y,refined_t_seq
)
starting_control_points_x,starting_control_points_y,starting_control_points_t,pixels_x,pixels_y,pixels_fg,binning_grid_x,binning_grid_y=batch_convert_to_ndarray(
    starting_control_points_x,starting_control_points_y,starting_control_points_t,pixels_x,pixels_y,pixels_fg,binning_grid_x,binning_grid_y
)

plt.imshow(img)
plt.scatter(optimized_control_points_x,optimized_control_points_y,marker='.',c='red')
plt.show()

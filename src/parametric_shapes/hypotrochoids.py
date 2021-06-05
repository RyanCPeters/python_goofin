import numba as nb
import numpy as np
from numba.typed import List as nbList
from numba import cuda
from math import cos, sin, pi,log
import cv2
from collections import namedtuple
from time import perf_counter
from functools import wraps
from src.parametric_shapes.diagnostics_gui import build_gui_window

tlen = 10000000
t_space = np.linspace(-600.*pi, 600. * pi, tlen, dtype=np.float64)
t_domain_str = f"t:[{str(t_space[0])}, {str(t_space[-1])}]"
int_min = np.iinfo(int).min
int_max = np.iinfo(int).max
# t_list_template = np.zeros((t_space.shape[0],2),int)
dlen = 200
rlen = 200
Rlen = 200
d_bounds = 0.1, 3.
r_bounds = .1,4.
d_space = np.linspace(d_bounds[0],d_bounds[1], dlen, dtype=np.float64)
# d_space = np.concatenate([d_space[::-1],d_space])
dlen = d_space.shape[0]
r_space = np.linspace(r_bounds[0],r_bounds[1] , rlen, dtype=np.float64)
# r_space = np.concatenate([r_space,r_space[::-1]])
rlen = r_space.shape[0]
R_space = np.linspace(.9, 3., Rlen, dtype=np.float64)
# R_space = np.concatenate([R_space[::-1],R_space])
Rlen = R_space.shape[0]

MIN_RGB_PALETTE_LEN = max(16384,Rlen*rlen*dlen)
uint16_info = np.iinfo(np.uint16)
uint16_max = uint16_info.max
color_palette = np.zeros((uint16_max*1, 3), np.uint16)
choices = [0,1,2,3]
choice_keys = "a","b","c","e","j","k","j2","i2"
choice_values = np.linspace(-100.,100.,2000,dtype=np.float64)
i2j2 = np.linspace(-10.,10.,2000,dtype=np.float64)
choice_and_ranges = {
    "j":[1.],
    "k":[1.],
    "a":[1.],
    "b":[1.],
    "c":[1.],
    "e":[1.],
    "j2":[1.],
    "i2":[1.],
    0:{
        "a":[v/10 for v in range(1,200)],
        "b":[v/10 for v in range(1,200)]
    },
    1:{
        "a": choice_values,
        "b": choice_values,
        "c": choice_values,
        "j": choice_values,
        "k": choice_values,
    },
    2:{
        "a": choice_values,
        "b": choice_values,
        "c": choice_values,
        "e": choice_values,
        "j2": i2j2,
        "i2": i2j2,
    },
    3:{
        # uses the R,r,d,t for basic hypotrochoid
    },
}
choice2equation = {
    0: {
        "x":"x = (a-b) * cos(t) + b * cos(t * ((_k) - 1))",
        "y":"y = (a-b) * sin(t) - b * sin(t * ((_k) - 1))",
        "params":(t_domain_str+"; a:{a}; b:{b}; _k:{_k}").split("; ")
    },
    1: {
        "x":"x = cos(a * t) - cos(b * t)**j",
        "y":"y = sin(c * t) - sin(d * t)**k",
        "params":(t_domain_str+"; a:{a}; b:{b}; c:{c}; d:{d}; j:{j}; k:{k}").split("; ")
    },
    2: {
        "x":"x = i2 * cos(a * t) - cos(b * t) * sin(c * t)",
        "y":"y = j2 * sin(d * t) - sin(e * t)",
        "params":(t_domain_str+"; a:{a}; b:{b}; c:{c}; d:{d}; e:{e}; i2:{i2}; j2:{j2}").split("; ")
    },
    3:{
        "x":"x = (R - r) * cos(t) + d * cos(t * (R - r) / r)",
        "y":"y = (R - r) * sin(t) - d * sin(t * (R - r) / r)",
        "params":(t_domain_str+"; R:{R};r:{r};d:{d}").split("; ")
    }
}

tbar_attr_fields = "R,r,d,a,b,c,e,j,k,j2,i2".split(",")
tbar_namedtuple = namedtuple("TBarAttrs", tbar_attr_fields)
tbar_keys = tbar_namedtuple(*tbar_attr_fields)
tbar_bounds = tbar_namedtuple(*((0, len(arr)-1) for arr in (R_space, r_space, d_space,
                                                            *(choice_values for _ in choice_keys[:-2]),
                                                            *(i2j2 for _ in choice_keys[-2:]))))
tbar_barnames = tbar_namedtuple(*(f"{k}:[{lo},{hi}]" for k,(lo,hi) in zip(tbar_bounds._fields,tbar_bounds)))
tbar_data = tbar_namedtuple(R_space,r_space,d_space,
                            *(choice_values for _ in choice_keys[:-2]),
                            *(i2j2 for _ in choice_keys[-2:]))


def findFontLocate(s_txt, h,w, font_face=cv2.FONT_HERSHEY_PLAIN, font_thick=1):
    """This function is adapted from a stack-overflow solution found at:

            https://stackoverflow.com/a/62996892/7412747
    """
    best_scale = 1.0
    bgd_w = w
    bgd_h = h
    txt_rect_w = 0
    txt_rect_h = 0
    baseline = 0
    for scale in np.arange(.1, 6., 0.001):
        (ret_w, ret_h), tmp_bsl = cv2.getTextSize(s_txt, font_face, scale, font_thick)
        tmp_w = ret_w + 2 * font_thick
        tmp_h = ret_h + 2 * font_thick + tmp_bsl
        if tmp_w >= bgd_w or tmp_h >= bgd_h:
            break
        else:
            baseline = tmp_bsl
            txt_rect_w = tmp_w
            txt_rect_h = tmp_h
            best_scale = scale
    lt_x, lt_y = round(bgd_w/2-txt_rect_w/2), round(bgd_h/2-txt_rect_h/2)
    rb_x, rb_y = round(bgd_w/2+txt_rect_w/2), round(bgd_h/2+txt_rect_h/2)-baseline
    return (lt_x, lt_y, rb_x, rb_y), best_scale, baseline


def rgb_color_palette(colors:np.ndarray):
    """Populates the input argument, colors, with a combination of 11 distinct colors. Where the palette starts with
    a one color and evenly fades to the next. The color sequence is roughly as follows:
        silver+red -> powder_blue -> silver -> pink -> blue -> green -> red -> bg -> silver -> rg -> gold_metalic
    colors must be at least 16384 3-channel elements long, or we get a ZeroDivisionError.
    """
    dtype_max = np.iinfo(colors.dtype).max
    red = np.array((50,50,200),np.float64)
    rg = np.array((100,100,25),np.float64)
    green = np.array((50,200,50),np.float64)
    bg = np.array((25,100,100),np.float64)
    blue = np.array((200,50,50),np.float64)
    br = np.array((100,25,100),np.float64)
    powder_blue = np.array((255,178,102),np.float64)
    pink = np.array((178,102,255),np.float64)
    gold_metalic = np.array((55,175,212),np.float64)
    silver = np.array((187,192,201),np.float64)
    if dtype_max>255:
        div_val = 1/255
        mult_val = div_val*dtype_max
        powder_blue *= mult_val
        pink *= mult_val
        gold_metalic *= mult_val
        silver *= mult_val
        red *= mult_val
        rg *= mult_val
        green *= mult_val
        bg *= mult_val
        blue *= mult_val
        br *= mult_val
    color_tuple =(((gold_metalic/2+green/8)/2,powder_blue),
                  (powder_blue,red),
                  (red, green),
                  (green,blue),
                  (blue, gold_metalic))
    regions = len(color_tuple)-1
    region_size = len(colors)//regions
    rem = len(colors)%region_size
    inv_reg_size = 1./region_size
    for _k in range(regions):
        k = _k*region_size
        c1, c2 = color_tuple[k%len(color_tuple)]
        color = c1[:]
        steps = c2-c1
        steps *= inv_reg_size
        stop = k + region_size
        for j in range(k,stop):
            colors[j] += np.round(color,0).astype(np.uint16)
            color += steps
    colors[-rem:] = colors[-1:-rem-1:-1]
    # from math import sqrt, ceil
    # demo_hw = int(ceil(sqrt(colors.shape[0])))
    # color_demo = np.zeros((demo_hw, demo_hw, 3), colors.dtype)
    # for y in range(demo_hw):
    #     for x in range(demo_hw):
    #         color_idx = (y * x)
    #         if color_idx < len(colors):
    #             color_demo[y, x] += colors[color_idx]
    # cv2.namedWindow("color_demo", cv2.WINDOW_NORMAL)
    # cv2.moveWindow("color_demo", 1, 1)
    # cv2.resizeWindow("color_demo", 1900, 1000)
    # cv2.imshow("color_demo", color_demo)
    # cv2.waitKey(0)
    # cv2.destroyWindow("color_demo")


rgb_color_palette(color_palette)


@cuda.jit(device=True)
def cuda_fun_shapes(choice,
                    R,r,d,t,
                    a,b,c,e,j,k,j2,i2):
    """
    * https://en.wikipedia.org/wiki/Parametric_equation#/media/File:Param_02.jpg
        x = (a - b) * cos(t) + b * cos(t * (k - 1))
        y = (a - b) * sin(t) - b * sin(t * (k - 1))
        k = a/b

    * https://en.wikipedia.org/wiki/Parametric_equation#/media/File:Param33_1.jpg
        x = cos( a*t) – cos(b*t)**j
        y = sin(c*t) – sin(d*t)**k

    * https://en.wikipedia.org/wiki/Parametric_equation#/media/File:Param_st_01.jpg
        x = j2 * cos(a*t) - cos(b*t) * sin(c*t)
        y = i2 * sin(d*t) - sin(e*t)

    :return: atbstracted mathematical y,x coordinates
    :rtype: Tuple[float,float]
    """
    if choice==0:
        _k = t*((a / b)-1.)
        ab = a-b
        x = ab * cos(t) + b * cos(_k)
        y = ab * sin(t) - b * sin(_k)
    elif choice==1:
        x = cos(a * t) - cos(b * t)**j
        y = sin(c * t) - sin(d * t)**k
    elif choice==2:
        x = j2 * cos(a*t) - cos(b*t) * sin(c*t)
        y = i2 * sin(d*t) - sin(e*t)
    else:
        y,x = cuda_basic_hyptrochoid(R,r,d,t)
    return y,x


@cuda.jit(device=True)
def cuda_basic_hyptrochoid(R:np.float64, r:np.float64, d:nb.float64, t:nb.float64):
    """
    A hypotrochoid is a curve traced by a point attached to a circle of radius r rolling around the inside of a fixed
    circle of radius R, where the point is at a distance d from the center of the interior circle.

    The parametric equations for the hypotrochoids are:
        x(t) = (R - r) * cos(t) + d * cos(t * (R - r) / r)
        y(t) = (R - r) * sin(t) - d * sin(t * (R - r) / r)
    """
    dr = R-r
    tdr = 1./r
    tdr *= dr
    tdr *= t
    x = dr * cos(t) + d * cos(tdr)
    y = dr * sin(t) - d * sin(tdr)
    return y,x


@cuda.jit#(fastmath=True)
def cuda_compute_update(arr:np.ndarray, t_arr:np.ndarray, color:np.ndarray,
                        choice:int,zoom:float,
                        R,r,d,a,b,c,e,j,k,j2,i2):
    """

    :param arr: the array with shape (h,w,channels) that we'll be displaying on screen
    :type arr: np.ndarray

    :param t_arr: an array of theta values. As we iterate over t_arr,
                  these values will be passed to the chosen function as t.
    :type t_arr: np.ndarray

    :param color: a 3 value array holding the B,G,R values for the locations mapped out by the chosen function.
    :type color: np.ndarray

    :param choice: chooses which function variant we use to compute x,y locations
    :type choice: np.float64

    :param R: used by choice 3;
                the outer circle's radius
    :type R: np.float64

    :param r: used by choice 3;
                the inner circle's radius
    :type r: np.float64

    :param d: used by choice 3;
                the distance of the projected point from the center of the inner circle
    :type d: np.float64

    :param j: used by choice 1; the power by which we scale the second term of x
    :type j: np.float64

    :param k: used by choice 1;
                the power by which we scale the second term of y
    :type k: np.float64

    :param a: used by choices 0, 1, and 2;
                a multiplicative factor for both x and y in choice 0
                a multiplicative factor for just x in choices 1, and 2.
    :type a: np.float64

    :param b: used by choices 0, 1, and 2;
                a multiplicative factor for both x and y in choice 0
                a multiplicative factor for just y in choices 1, and 2.
    :type b: np.float64

    :param c: used by choices 1, and 2;
                a multiplicative factor for y
    :type c: np.float64

    :param e: used by choice 2;
                a multiplicative factor for y
    :type e: np.float64

    :param j2:  used by choice 2;
                a multiplicative factor for the first term in x
    :type j2: np.float64

    :param i2: used by choice 2;
                a multiplicative factor for the first term in y
    :type i2: np.float64

    :return:
    :rtype:
    """
    ti = cuda.grid(1)
    istep = cuda.gridsize(1)
    h,w = arr.shape[:2]
    fh = h * .95
    fw = w * .8
    h5 = fh * .5 + h * .025
    w5 = fw * .5 + w * .1
    y_max = x_max = zoom
    y_min = -y_max
    x_min = -x_max
    yspan = y_max-y_min
    xspan = x_max-x_min
    fh_span = fh*(1./yspan)
    fw_span = fw*(1./xspan)
    for i in range(0,t_arr.shape[0],istep):
        i += ti
        if i<t_arr.shape[0]:
            y,x = cuda_fun_shapes(choice,R,r,d,t_arr[i],a,b,c,e,j,k,j2,i2)
            iy = round(y*fh_span+h5)
            ix = round(x*fw_span+w5)
        cuda.syncthreads()
        if i<t_arr.shape[0]:
            if 0<=iy<arr.shape[0] and 0<=ix<arr.shape[1]:
                for c in range(arr.shape[2]):
                    arr[iy,ix,c] = color[c]
        cuda.syncthreads()


@nb.njit(parallel=True,nogil=True)
def update_arr(arr:np.ndarray,t_list:list,color:np.ndarray):
    lst_len = len(t_list)
    blocks = 20
    block_size = (lst_len+blocks-1)//blocks
    for block in nb.prange(blocks):
        strt = block*block_size
        stop = min(strt+block_size,lst_len)
        for i in range(strt,stop):
            y,x = t_list[i]
            yl = max(0,y-1)
            yh = min(arr.shape[0],y+1)
            xl = max(0,x-1)
            xh = min(arr.shape[1],x+1)
            for _y in range(yl,yh):
                for _x in range(xl,xh):
                    arr[_y,_x] = color


@cuda.jit(fastmath=True)
def cuda_update_arr(arr:np.ndarray, t_list:np.ndarray, color:np.ndarray):
    ti = cuda.grid(1)
    istep = cuda.gridsize(1)
    for _i in range(0,t_list.shape[0],istep):
        i = _i+ti
        if i<t_list.shape[0]:
            y, x = t_list[i]
        cuda.syncthreads()
        if i<t_list.shape[0]:
            yl = max(0, y - 1)
            yh = min(arr.shape[0], y + 1)
            xl = max(0, x - 1)
            xh = min(arr.shape[1], x + 1)
            for _y in range(yl, yh):
                for _x in range(xl, xh):
                    for j in range(color.shape[0]):
                        arr[_y, _x,j] = color[j]
        cuda.syncthreads()


@cuda.jit(fastmath=True)
def cuda_zero_arr(arr:np.ndarray):
    ti,tj = cuda.grid(2)
    istep,jstep = cuda.gridsize(2)
    for i in range(ti,arr.shape[0],istep):
        for j in range(tj,arr.shape[1],jstep):
            for c in range(arr.shape[2]):
                arr[i,j,c] *= 0


def get_updater(bpg,tpb,stream):
    @wraps(cuda_compute_update)
    def wrapper(*args,**kwargs):
        return cuda_compute_update[bpg,tpb,stream](*args,**kwargs)
    return wrapper


def cuda_inspect_hypotrochoid():
    c_pos = 0
    c_step = 1
    tpb2d = 16,16
    tpb1d = 64
    font_thick = 4
    font_color = uint16_max-1,uint16_max-1,uint16_max-1
    intended_height = 75
    def update_iter_pos():
        nonlocal c_pos,c_step, pos,pos_dir
        if c_step>0:
            if c_pos+c_step>=len(color_palette):
                c_step = -1
        elif c_pos+c_step<0:
            c_step = 1
        if pos_dir>0:
            if pos+pos_dir>pos_max:
                pos_dir = -1
        elif pos+pos_dir<0:
            pos_dir = 1
        c_pos += c_step
        pos += pos_dir
        return

    def put_text(ss, y_offset=0,**kwargs):
        nonlocal port
        if isinstance(ss,str):
            ss = [ss]
        for s in ss:
            s = s.format(**kwargs)
            (lt_x, lt_y, rb_x, rb_y), best_scale, baseline = findFontLocate(s, intended_height, w, font_thick=font_thick)
            cv2.putText(port, s, (10, rb_y + y_offset), cv2.FONT_HERSHEY_PLAIN,
                        best_scale, font_color, font_thick, cv2.LINE_AA)
            y_offset += intended_height
        return

    h,w = 2000,3000
    winname = "window"
    cv2.namedWindow(winname,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(winname,1700,900)
    cv2.moveWindow(winname,10,10)

    port = np.zeros((h,w,3),np.uint16)
    d_port = cuda.to_device(port)
    d_colors = cuda.to_device(color_palette)
    d_tarr = cuda.to_device(t_space)
    bpg2d = (port.shape[0]//4+tpb2d[0]-1)//tpb2d[0],(port.shape[1]//4+tpb2d[1]-1)//tpb2d[1]
    bpg1d = (t_space.shape[0]//4+tpb1d-1)//tpb1d
    stream = cuda.stream()
    # updater = cuda_compute_update[bpg1d, tpb1d, stream]
    updater = get_updater(bpg1d, tpb1d, stream)
    zero_out = cuda_zero_arr[bpg2d,tpb2d,stream]
    k = -1
    pos_dir = 1
    pos_max = d_space.shape[0]-1
    pos = pos_max
    while k!=27:
        for R in R_space:
            for r in r_space:
                for _ in range(pos_max):
                    for choice in choices:
                        choice_selected_params = {k:choice_and_ranges[choice].get(k,choice_and_ranges[k]) for k in choice_keys}
                        choice_text_d = choice2equation[choice]
                        for j in choice_selected_params["j"]:
                            d = d_space[pos]
                            for k in choice_selected_params["k"]:
                                for a in choice_selected_params["a"]:
                                    for b in choice_selected_params["b"]:
                                        _k = a/b
                                        for c in choice_selected_params["c"]:
                                            for e in choice_selected_params["e"]:
                                                for j2 in choice_selected_params["j2"]:
                                                    for i2 in choice_selected_params["i2"]:
                                                        updater(d_port, d_tarr, d_colors[c_pos],choice,R,r,d,j,k,a,b,c,e,j2,i2)
                                                        d_port.copy_to_host(port,stream)
                                                        put_text(choice_text_d['x'])
                                                        put_text(choice_text_d['y'],intended_height)
                                                        put_text(choice_text_d['params'],intended_height+intended_height,a=a,b=b,c=c,d=d,e=e,j=j,k=k,j2=j2,i2=i2,_k=_k)
                                                        zero_out(d_port)
                                                        cv2.imshow(winname,port)
                                                        k=cv2.waitKey(1)
                    update_iter_pos()
                    if k==27:
                        break
                if k == 27:
                    break
            if k==27:
                break
    cv2.destroyWindow(winname)

def trackbar_update_wrapper(trackbar_values:dict,key:str):
    def inner(val):
        nonlocal trackbar_values
        trackbar_values[key] = val
    return inner

def cuda_inspect_hypotrochoid_sliders():
    tpb2d = 16,16
    tpb1d = 64
    font_thick = 4
    available_func_count = 4
    trackbar_values = {
        "c_pos":0,
        "choice":0,
    }
    trackbar_values.update(((k,bound[0]) for k,bound in zip(tbar_keys,tbar_bounds)))
    font_color = uint16_max-1,uint16_max-1,uint16_max-1
    intended_height = 75
    color_tbar_name = f"color: [0, {color_palette.shape[0]-1}]"
    func_tbar_name = f"choice: [0, {available_func_count}]"

    def put_text(ss, y_offset=0,**kwargs):
        nonlocal arr
        if isinstance(ss,str):
            ss = [ss]
        s = ss[0]
        (lt_x, lt_y, rb_x, rb_y), best_scale, baseline = findFontLocate(s, intended_height, w, font_thick=font_thick)
        cv2.putText(arr, s, (10, rb_y + y_offset), cv2.FONT_HERSHEY_PLAIN,
                    best_scale, font_color, font_thick, cv2.LINE_AA)
        y_offset += intended_height
        for s in ss[1:]:
            s = s.format(**kwargs)
            (lt_x, lt_y, rb_x, rb_y), best_scale, baseline = findFontLocate(s, intended_height, w, font_thick=font_thick)
            cv2.putText(arr, s, (10, rb_y + y_offset), cv2.FONT_HERSHEY_PLAIN,
                        best_scale, font_color, font_thick, cv2.LINE_AA)
            y_offset += intended_height
        return

    h,w = 2000,3500
    winname = "window"
    cv2.namedWindow(winname,cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow(winname,1700,900)
    cv2.moveWindow(winname,10,10)
    cv2.createTrackbar(color_tbar_name,"",
                       trackbar_values["c_pos"],color_palette.shape[0]-1,
                       lambda val: trackbar_values.update({"c_pos":val}))
    cv2.createTrackbar(func_tbar_name,"",
                       trackbar_values["choice"],available_func_count-1,
                       lambda val: trackbar_values.update({"choice":val}))
    for name,bounds,key in zip(tbar_barnames,tbar_bounds,tbar_keys):
        cv2.createTrackbar(name,"",bounds[0],bounds[1],trackbar_update_wrapper(trackbar_values,key))

    arr = np.zeros((h,w,3),np.uint16)
    d_arr = cuda.to_device(arr)
    d_colors = cuda.to_device(color_palette)
    d_tarr = cuda.to_device(t_space)
    bpg2d = (arr.shape[0]//4+tpb2d[0]-1)//tpb2d[0],(arr.shape[1]//4+tpb2d[1]-1)//tpb2d[1]
    bpg1d = (t_space.shape[0]//100+tpb1d-1)//tpb1d
    stream = cuda.stream()
    k = -1
    while k!=27:
        choice = trackbar_values["choice"]
        choice_text_d = choice2equation[choice]
        _params = tuple((k,v[trackbar_values.get(k,0)]) for k,v in zip(tbar_data._fields,tbar_data))
        args = (tpl[1] for tpl in _params)
        cuda_compute_update[bpg1d,tpb1d,stream](d_arr, d_tarr, d_colors[trackbar_values["c_pos"]], choice,*args)
        stream.synchronize()
        d_arr.copy_to_host(arr,stream)
        stream.synchronize()
        cuda_zero_arr[bpg2d,tpb2d,stream](d_arr)
        put_text(choice_text_d['x'])
        put_text(choice_text_d['y'],intended_height)
        params = {k:v for k,v in _params}
        params["_k"] = params["a"]/params["b"]
        put_text(choice_text_d['params'],intended_height+intended_height,**params)
        cv2.imshow(winname,arr)
        k=cv2.waitKey(25)

    cv2.destroyWindow(winname)

def cuda_inspect_hypotroicoid_gui():
    tpb2d = 16, 16
    tpb1d = 64
    font_thick = 4
    available_func_count = 4
    persistent_values = {
        "c_pos": 0,
        "choice": 0,
        "zoom":1.,
    }
    persistent_values.update(((k, bound[0]) for k, bound in zip(tbar_keys, tbar_bounds)))
    font_color = uint16_max - 1, uint16_max - 1, uint16_max - 1
    intended_height = 75

    def put_text(ss, y_offset=0, **kwargs):
        nonlocal arr
        if isinstance(ss, str):
            ss = [ss]
        s = ss[0]
        (lt_x, lt_y, rb_x, rb_y), best_scale, baseline = findFontLocate(s, intended_height, w, font_thick=font_thick)
        cv2.putText(arr, s, (10, rb_y + y_offset), cv2.FONT_HERSHEY_PLAIN,
                    best_scale, font_color, font_thick, cv2.LINE_AA)
        y_offset += intended_height
        for s in ss[1:]:
            s = s.format(**kwargs)
            (lt_x, lt_y, rb_x, rb_y), best_scale, baseline = findFontLocate(s, intended_height, w,
                                                                            font_thick=font_thick)
            cv2.putText(arr, s, (10, rb_y + y_offset), cv2.FONT_HERSHEY_PLAIN,
                        best_scale, font_color, font_thick, cv2.LINE_AA)
            y_offset += intended_height
        return

    h, w = 2000, 3250
    winname = "window"
    cv2.namedWindow(winname, cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow(winname, 1700, 900)
    cv2.moveWindow(winname, 10, 10)
    arr = np.zeros((h, w, 3), np.uint16)
    d_arr = cuda.to_device(arr)
    d_colors = cuda.to_device(color_palette)
    d_tarr = cuda.to_device(t_space)
    bpg2d = (arr.shape[0] // 4 + tpb2d[0] - 1) // tpb2d[0], (arr.shape[1] // 4 + tpb2d[1] - 1) // tpb2d[1]
    bpg1d = (t_space.shape[0] // 100 + tpb1d - 1) // tpb1d
    stream = cuda.stream()
    window = build_gui_window()
    try:
        event = ""
        while event is not None and event != "Cancel":
            event, values = window.read()
            values = {k:float(v) for k,v in values.items() if v!=""}
            persistent_values.update(values)
            choice = int(persistent_values["choice"])
            c_pos = int(persistent_values["c_pos"])
            zoom = persistent_values["zoom"]
            choice_text_d = choice2equation[choice]
            _params = tuple((k, persistent_values.get(k, 1.)) for k in tbar_attr_fields)
            args = (tpl[1] for tpl in _params)
            cuda_compute_update[bpg1d, tpb1d, stream](d_arr, d_tarr, d_colors[c_pos], choice, zoom, *args)
            stream.synchronize()
            d_arr.copy_to_host(arr, stream)
            stream.synchronize()
            cuda_zero_arr[bpg2d, tpb2d, stream](d_arr)
            put_text(choice_text_d['x'])
            put_text(choice_text_d['y'], intended_height)
            params = {k: v for k, v in _params}
            if params["b"]==0:
                params["_k"] = (params["a"]+.000000001)/(params["b"]+.000000001)
            else:
                params["_k"] = params["a"] / params["b"]
            put_text(choice_text_d['params'], intended_height + intended_height, **params)
            cv2.imshow(winname, arr)
            cv2.waitKey(1)

        cv2.destroyWindow(winname)

    finally:
        window.close()


if __name__ == '__main__':
    # inspect_hypotrochoid()
    # cuda_inspect_hypotrochoid()
    # cuda_inspect_hypotrochoid_sliders()
    cuda_inspect_hypotroicoid_gui()

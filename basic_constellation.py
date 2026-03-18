# from https://github.com/plasma-observatory/po-ephemeris/ without vtk

import numpy as np
import os

DEFAULT_FILE_PATH = "./PO_constellation"
float_fmt = "%20.16e"

def clean(FILE_PATH = DEFAULT_FILE_PATH):
    try:
        os.remove(FILE_PATH + ".txt")
    except:
        pass

def export_np(x,y,z,vx,vy,vz, FILE_PATH=DEFAULT_FILE_PATH):
    
    pts_array = np.vstack([np.array([1,2,3,4,5,6,7]),x,y,z,vx,vy,vz]).T
    np.savetxt(FILE_PATH+".txt", pts_array, fmt="%d"+6*(", "+float_fmt))

def import_np(FILE_PATH=DEFAULT_FILE_PATH):
    data = np.loadtxt(FILE_PATH+".txt", delimiter=',')
    print(data)
    return data

def run(outerscale = 5000e3, FILE_PATH=DEFAULT_FILE_PATH):

    # Define vertices
    x = np.zeros(7)
    y = np.zeros(7)
    z = np.zeros(7)

    # inner tetrahedron size factor; required: < 1/4
    f=1.0/5.0

    x[0], y[0], z[0] = 0.0, 0.0, 0.0
    x[1], y[1], z[1] = 1.0, 1.0, 0.0
    x[2], y[2], z[2] = 1.0, 0.0, 1.0
    x[3], y[3], z[3] = 0.0, 1.0, 1.0
    x[4], y[4], z[4] = 1.0, 1.0, 0.0
    x[5], y[5], z[5] = 1.0, 0.0, 1.0
    x[6], y[6], z[6] = 0.0, 1.0, 1.0

    normf = np.sqrt((x[0]-x[1])**2 + (y[0]-y[1])**2 + (z[0]-z[1])**2)

    x /= normf
    y /= normf
    z /= normf

    x[4:] *= f
    y[4:] *= f
    z[4:] *= f

    x *= outerscale
    y *= outerscale
    z *= outerscale

    # dummy velocities
    vx = np.zeros(7)
    vy = np.zeros(7)
    vz = np.zeros(7)

    # This concludes setting the coordinates. Now these will be saved to various formats.
    try:
        export_np(x,y,z,vx,vy,vz, FILE_PATH=DEFAULT_FILE_PATH)
    except Exception as e:
        print("Could not export in a numpy format, error was "+ str(e))
        raise e

def fly(constellation = None, FILE_PATH=DEFAULT_FILE_PATH, suffix="_flight",start=[8.5e7,-1e7,-1e6], end=[5.5e7,-1e7,-1e6], steps=400, time_over_segment=3600):
    if constellation is None:
        constellation = import_np(FILE_PATH)

    deltas = np.linspace(start,end,steps)
    print(constellation)
    print("start", np.array(start)/1e7, "end", np.array(end)/1e7)
    constellation_points = np.tile(constellation,(steps,1))

    # print(constellation_points[:20,:])
    vx = (end[0]-start[0])/time_over_segment
    vy = (end[1]-start[1])/time_over_segment
    vz = (end[2]-start[2])/time_over_segment

    path = np.zeros((steps*7,8))
    for i in [0,1,2,3,4,5,6]:
        path[i::7,2:5] = constellation_points[i::7,1:4]+deltas
        path[i::7,1] = constellation_points[i::7,0]
        path[i::7,0] = range(steps)
    path[:,5] = vx
    path[:,6] = vy
    path[:,7] = vz

    # np.savetxt(FILE_PATH+suffix+".txt", path, header="time_index, spacecraft_id, x, y, z, vx, vy, vz", fmt="%d, %d"+6*(", "+float_fmt))

    return path



if __name__ == "__main__":
    FILE_PATH=DEFAULT_FILE_PATH
    run(FILE_PATH=DEFAULT_FILE_PATH)
    fly(FILE_PATH=DEFAULT_FILE_PATH)

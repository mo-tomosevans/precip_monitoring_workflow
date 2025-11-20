#!/usr/bin/env python3

import iris
import logging
import argparse
from datetime import datetime, timedelta

logging.basicConfig()
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datetime", required=True)
    parser.add_argument("--datadir", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    return args

def check_bounds(timeslice, hour):
    # return a True if bounds are correct for given length of time
    bounds = timeslice.coord("time").bounds[0]
    assert bounds[1] - bounds[0] == hour

def create_total_cube(filepath, precip_type):
    # create cube that combines rain and snow stash from filepath
    if precip_type == "convective":
        STASH = ["m01s05i201", "m01s05i202"]
    elif precip_type == "large_scale":
        STASH = ["m01s04i201", "m01s04i202"]
    else:
        raise ValueError(f"Invalid precip_type: {precip_type}")

    rain = iris.load_cube(filepath, iris.AttributeConstraint(STASH=STASH[0]))
    snow = iris.load_cube(filepath, iris.AttributeConstraint(STASH=STASH[1]))

    total_precip = rain + snow
    total_precip.rename(f"total_{precip_type}_precipitation")

    return total_precip



def main():
    args = parse_args()
    dt = args.datetime
    datadir = args.datadir
    output_dir = args.outdir
    LOGGER.info(f" DATETIME: {dt}")
    LOGGER.info(f" DATADIR: {datadir}")
    LOGGER.info(f" OUTDIR: {output_dir}")


    T0 = f"{datadir}/{dt}_gl-mn_000.pp"
    T6 = f"{datadir}/{dt}_gl-mn_006.pp"
    

    ############### CONVECTIVE ANALYSIS ################
    conv_cube_t0 = create_total_cube(T0, "convective")
    conv_cube_t6 = create_total_cube(T6, "convective")
    conv_t0_time_slices = iris.cube.CubeList(conv_cube_t0.slices_over(["time"]))
    conv_t6_time_slices = iris.cube.CubeList(conv_cube_t6.slices_over(["time"]))
    
    # T0 correct bounds is only slice
    assert(len(conv_t0_time_slices) == 1)
    conv_t0_analysis = conv_t0_time_slices[0]

    # T6 correct bounds are last timeslice of *_006.pp file
    conv_t6_analysis = conv_t6_time_slices[-1]

    # assert timeslice is correct
    check_bounds(conv_t0_analysis, 3.0)
    check_bounds(conv_t6_analysis, 9.0)
    LOGGER.info("Bounds are correct, 6hr difference")

    conv_analysis_cube = conv_t6_analysis - conv_t0_analysis
    conv_analysis_cube.rename("(t+6)-(t+0) conv analysis")
    LOGGER.info(f"conv analysis cube: {conv_analysis_cube}")
    
    # analysis VT will be DT+6
    dt_object = datetime.strptime(dt, "%Y%m%dT%H%MZ")
    vt_object = dt_object + timedelta(hours=6)
    VT = vt_object.strftime("%Y%m%d%H%MZ")

    analysis_path_to_save = f"{output_dir}/{dt}_VT{VT}_conv_analysis.nc"

   # iris.save(conv_analysis_cube, analysis_path_to_save)



    ############### LARGE SCALE ANALYSIS ################
    lsr_cube_t0 = create_total_cube(T0, "large_scale")
    lsr_cube_t6 = create_total_cube(T6, "large_scale")
    lsr_t0_time_slices = iris.cube.CubeList(lsr_cube_t0.slices_over(["time"]))
    lsr_t6_time_slices = iris.cube.CubeList(lsr_cube_t6.slices_over(["time"]))
    
    # T0 correct bounds is only slice
    assert(len(lsr_t0_time_slices) == 1)
    lsr_t0_analysis = lsr_t0_time_slices[0]

    # T6 correct bounds are last timeslice of *_006.pp file
    lsr_t6_analysis = lsr_t6_time_slices[-1]

    # assert timeslice is correct
    check_bounds(lsr_t0_analysis, 3.0)
    check_bounds(lsr_t6_analysis, 9.0)
    LOGGER.info("Bounds are correct, 6hr difference")

    lsr_analysis_cube = lsr_t6_analysis - lsr_t0_analysis
    lsr_analysis_cube.rename("(t+6)-(t+0) large scale analysis")
    LOGGER.info(f"large scale analysis cube: {lsr_analysis_cube}")
    
    # analysis VT will be DT+6
    dt_object = datetime.strptime(dt, "%Y%m%dT%H%MZ")
    vt_object = dt_object + timedelta(hours=6)
    VT = vt_object.strftime("%Y%m%dT%H%MZ")

    lsr_analysis_path_to_save = f"{output_dir}/{dt}_VT{VT}_lsr_analysis.nc"
   # iris.save(lsr_analysis_cube, lsr_analysis_path_to_save)

    total_analysis = lsr_analysis_cube.copy()
    total_analysis.long_name = "Total_Precip_Accumulation"
    total_data = lsr_analysis_cube.data + conv_analysis_cube.data   
    total_analysis.data = total_data

    accum = 6

    init_time = datetime.strptime(dt, "%Y%m%dT%H%MZ")  # adjust format as needed
    lead_hours = accum  # or use your lead time variable
    valid_time = init_time + timedelta(hours=accum)

    total_analysis.attributes['valid_time'] = valid_time.strftime("%Y%m%dT%H%MZ")
    total_path_to_save = f"{output_dir}/{dt}_VT{VT}_analysis.nc"

    iris.save(total_analysis, total_path_to_save)


if __name__ == "__main__":
    main()
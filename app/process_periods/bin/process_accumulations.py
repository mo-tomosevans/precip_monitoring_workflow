#!/usr/bin/env python3

import iris
import logging
import oemplotlib
from oemplotlib.cube_utils import running_accum_to_period, fix_running_cube_time, separate_realization_time
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
    parser.add_argument("--trial1")
    parser.add_argument("--trial2")
    args = parser.parse_args()
    return args


class Precip_accumulations_large_scale:
    # class level variable
    # large scale rain stash, large scale snow stash
    rainfall_stash = ["m01s04i201", "m01s04i202"]

    def __init__(self, filepath):
        self.filepath = filepath

    def get_cube_accumulations(self, period_hrs):
        cubes = iris.load(self.filepath)

        def _loader(cubelist, stash_constraint):
            try:
                constraint = iris.Constraint(cube_func=lambda cube: cube.cell_methods)
                outcube = cubelist.extract_cube(
                    iris.AttributeConstraint(STASH=stash_constraint) & constraint,
                )
            except iris.exceptions.ConstraintMismatchError as err:
                outcube = cubelist.extract_cube(
                    iris.AttributeConstraint(STASH=stash_constraint)
                )
                msg = ""
                for cube in outcube:
                    msg += f"{cube}\n"
                LOGGER.exception(
                    "error loading for stash %s, found cubes:\n%s",
                    stash_constraint,
                    msg,
                )
                raise err
            return outcube.copy()

        stratiform_rain = _loader(cubes, Precip_accumulations_large_scale.rainfall_stash[0])
        try:
            stratiform_snow = _loader(cubes, Precip_accumulations_large_scale.rainfall_stash[1])
            rainsnow = stratiform_rain + stratiform_snow
        except iris.exceptions.ConstraintMismatchError:
            LOGGER.warning(
                "plot_total_precip_rate couldn't load stratiform snow rates, "
                "plotting will resume assuming this is a startiform rain-only model"
            )
            rainsnow = stratiform_rain

        # first handle ensemble possibly having multidimensional time coordinates
        rainsnow_separated = separate_realization_time(rainsnow)

        LOGGER.info(f"rainsnow_separated time: {rainsnow_separated.coord('time')}")

        # only want hourly output so extract times that are on the hour
        rainsnow_filtered = iris.cube.CubeList()
        if isinstance(rainsnow_separated, iris.cube.CubeList):
            for p in rainsnow_separated:
                try:
                    p = oemplotlib.cube_utils.snap_to_time(
                        p,
                        minutes_past_hour=0,
                        max_window_minutes=5,
                    )
                    rainsnow_filtered.append(p)
                except ValueError:
                    # probably a sub-hourly off hour cube
                    pass
        else:
            rainsnow_filtered.append(
                oemplotlib.cube_utils.snap_to_time(
                    rainsnow_separated,
                    minutes_past_hour=0,
                    max_window_minutes=5,
                )
            )
        if len(rainsnow_filtered) < 1:
            raise ValueError("Unable to separate/snap rainsnow cubes")
        elif len(rainsnow_filtered) == 1:
            rainsnow = rainsnow_filtered[0]
        else:
            rainsnow = rainsnow_filtered.merge()

        LOGGER.info(f"rainsnow time coord: {rainsnow.coord('time')}")
        rainsnow = fix_running_cube_time(rainsnow)

        def _add_cell_methods(cube):
            for method in stratiform_rain.cell_methods:
                if not any(m == method for m in cube.cell_methods):
                    cube.add_cell_method(method)


        if isinstance(rainsnow, iris.cube.CubeList):
            [p.attributes.update(stratiform_rain.attributes) for p in rainsnow]
            # kg m-2 is equivalent to mm as density of water is approx 1000 kg m-3
            [p.convert_units("kg m-2") for p in rainsnow]
            [_add_cell_methods(p) for p in rainsnow]
        else:
            rainsnow.attributes.update(stratiform_rain.attributes)
            rainsnow.convert_units("kg m-2")
            _add_cell_methods(rainsnow)


        LOGGER.info(f"RAINSNOW: {rainsnow}")
        LOGGER.info(f"RAINSNOW coord time: {rainsnow.coord('time')}")
        #LOGGER.info(f"RAINSNOW time points: {rainsnow[0].points}")

        try:
            accum_rain = oemplotlib.cube_utils.running_accum_to_period(
                rainsnow,
                period_minutes=period_hrs * 60,
                out_cube_name=f"Rain Accumulation {period_hrs} hrly",
            )
        except Exception:
            LOGGER.exception(
                "plot_rain_amnt: Error accumulating rain for %s period, skipping",
                period_hrs,
            )

        return accum_rain




class Precip_accumulations_convective:
    # class level variable
    # convective rain stash, convective snow stash
    rainfall_stash = ["m01s05i201", "m01s05i202"]

    def __init__(self, filepath):
        self.filepath = filepath

    def get_cube_accumulations(self, period_hrs):
            cubes = iris.load(self.filepath)

            def _loader(cubelist, stash_constraint):
                try:
                    constraint = iris.Constraint(cube_func=lambda cube: cube.cell_methods)
                    outcube = cubelist.extract_cube(
                        iris.AttributeConstraint(STASH=stash_constraint) & constraint,
                    ) 
                except iris.exceptions.ConstraintMismatchError as err:
                    outcube = cubelist.extract_cube(
                        iris.AttributeConstraint(STASH=stash_constraint)
                    )
                    msg = ""
                    for cube in outcube:
                        msg += f"{cube}\n"
                    LOGGER.exception(
                        "error loading for stash %s, found cubes:\n%s",
                        stash_constraint,
                        msg,
                    )
                    raise err
                return outcube.copy()

            stratiform_rain = _loader(cubes, Precip_accumulations_convective.rainfall_stash[0])
            try:
                stratiform_snow = _loader(cubes, Precip_accumulations_convective.rainfall_stash[1])
                rainsnow = stratiform_rain + stratiform_snow
            except iris.exceptions.ConstraintMismatchError:
                LOGGER.warning(
                    "plot_total_precip_rate couldn't load stratiform snow rates, "
                    "plotting will resume assuming this is a startiform rain-only model"
                )
                rainsnow = stratiform_rain
            # first handle ensemble possibly having multidimensional time coordinates
            rainsnow_separated = oemplotlib.cube_utils.separate_realization_time(rainsnow)
            # only want hourly output so extract times that are on the hour
            rainsnow_filtered = iris.cube.CubeList()
            if isinstance(rainsnow_separated, iris.cube.CubeList):
                for p in rainsnow_separated:
                    try:
                        p = oemplotlib.cube_utils.snap_to_time(
                            p,
                            minutes_past_hour=0,
                            max_window_minutes=5,
                        )
                        rainsnow_filtered.append(p)
                    except ValueError:
                        # probably a sub-hourly off hour cube
                        pass
            else:
                rainsnow_filtered.append(
                    oemplotlib.cube_utils.snap_to_time(
                        rainsnow_separated,
                        minutes_past_hour=0,
                        max_window_minutes=5,
                    )
                )
            if len(rainsnow_filtered) < 1:
                raise ValueError("Unable to separate/snap rainsnow cubes")
            elif len(rainsnow_filtered) == 1:
                rainsnow = rainsnow_filtered[0]
            else:
                rainsnow = rainsnow_filtered.merge()
                
            rainsnow = fix_running_cube_time(rainsnow)
            if isinstance(rainsnow, iris.cube.CubeList):
                [p.attributes.update(stratiform_rain.attributes) for p in rainsnow]
                # kg m-2 is equivalent to mm as density of water is approx 1000 kg m-3
                [p.convert_units("kg m-2") for p in rainsnow]
                for method in stratiform_rain.cell_methods:
                    [p.add_cell_method(method) for p in rainsnow]
            else:
                rainsnow.attributes.update(stratiform_rain.attributes)
                rainsnow.convert_units("kg m-2")
                for method in stratiform_rain.cell_methods:
                    rainsnow.add_cell_method(method)

            try:
                accum_rain = oemplotlib.cube_utils.running_accum_to_period(
                    rainsnow,
                    period_minutes=period_hrs * 60,
                    out_cube_name=f"Rain Accumulation {period_hrs} hrly",
                )
            except Exception:
                LOGGER.exception(
                    "plot_rain_amnt: Error accumulating rain for %s period, skipping",
                    period_hrs,
                )

            return accum_rain
    
def get_hrly_total_precip_amnt(
    cubes, large_rain_stash, large_snow_stash, conv_rain_stash, conv_snow_stash
):
    def _loader(cubelist, stash_constraint):
        try:
            constraint = iris.Constraint(cube_func=lambda cube: cube.cell_methods)
            outcube = cubelist.extract_cube(
                iris.AttributeConstraint(STASH=stash_constraint) & constraint,
            )
        except iris.exceptions.ConstraintMismatchError as err:
            outcube = cubelist.extract(iris.AttributeConstraint(STASH=stash_constraint))
            msg = ""
            for cube in outcube:
                msg += f"{cube}\n"
            LOGGER.exception(
                "error loading for stash %s, found cubes:\n%s",
                stash_constraint,
                msg,
            )
            raise err
        return outcube.copy()

    large_rain = _loader(cubes, large_rain_stash)
    large_snow = _loader(cubes, large_snow_stash)

    try:
        conv_rain = _loader(cubes, conv_rain_stash)
        conv_snow = _loader(cubes, conv_snow_stash)
        total_rain = large_rain + conv_rain
        total_snow = large_snow + conv_snow
    except iris.exceptions.ConstraintMismatchError:
        LOGGER.warning(
            "plot_total_precip_rate couldn't load convective precip amounts, "
            "plotting will resume assuming this is a convection permitting model"
        )
        total_rain = large_rain
        total_snow = large_snow

    precip = total_rain + total_snow

    # first handle ensemble possibly having multi-dimensional time coordinates
    precip_separated = oemplotlib.cube_utils.separate_realization_time(precip)

    # only want hourly output so extract times that are on the hour
    precip_filtered = iris.cube.CubeList()
    if isinstance(precip_separated, iris.cube.CubeList):
        for p in precip_separated:
            try:
                p = oemplotlib.cube_utils.snap_to_time(
                    p,
                    minutes_past_hour=0,
                    max_window_minutes=5,
                )
                precip_filtered.append(p)
            except ValueError:
                # probably a sub-hourly off hour cube
                pass
    else:
        precip_filtered.append(
            oemplotlib.cube_utils.snap_to_time(
                precip_separated,
                minutes_past_hour=0,
                max_window_minutes=5,
            )
        )
    if len(precip_filtered) < 1:
        raise ValueError("Unable to separate/snap precip cubes")
    elif len(precip_filtered) == 1:
        precip = precip_filtered[0]
    else:
        precip = precip_filtered.merge()

    precip = oemplotlib.cube_utils.fix_running_cube_time(precip)

    if isinstance(precip, iris.cube.CubeList):
        [p.attributes.update(large_rain.attributes) for p in precip]
        # kg m-2 is equivalent to mm as density of water is approx 1000 kg m-3
        [p.convert_units("kg m-2") for p in precip]
        for method in large_rain.cell_methods:
            [p.add_cell_method(method) for p in precip]
    else:
        precip.attributes.update(large_rain.attributes)
        precip.convert_units("kg m-2")
        for method in large_rain.cell_methods:
            precip.add_cell_method(method)

    return precip



def main():
    args = parse_args()
    dt = args.datetime
    datadir = args.datadir
    output_dir = args.outdir
    LOGGER.info(f" DATETIME: {dt}")
    LOGGER.info(f" DATADIR: {datadir}")
    LOGGER.info(f" OUTDIR: {output_dir}")
    
    accum_periods = [6, 24]

    for accum in accum_periods:

        val = accum 
        file = f"{datadir}/{accum}_hour/{dt}_gl-mn_{accum:03d}.pp"
        #print(f"This is file_to_read: {filepath}")

        #cubes_to_read = iris.load(filepath)
        #for file in cubes_to_read:
        #print(f" This is file being processed: {file}")
        lsr_precip = Precip_accumulations_large_scale(file)
        print(lsr_precip)
        conv_precip = Precip_accumulations_convective(file)
        print(conv_precip)
        
        LOGGER.info(f"accumulation period: {accum}. Making accumulations..")
        lsr_accumulations = lsr_precip.get_cube_accumulations(accum)
        conv_accumulations = conv_precip.get_cube_accumulations(accum)

        total_accumulations = lsr_accumulations.copy()
        total_accumulations.long_name = "Total_Precip_Accumulation"
        
        total_data = lsr_accumulations.data + conv_accumulations.data
        total_accumulations.data = total_data

        init_time = datetime.strptime(dt, "%Y%m%dT%H%MZ")  # adjust format as needed
        lead_hours = accum  # or use your lead time variable
        valid_time = init_time + timedelta(hours=accum)

        total_accumulations.attributes['valid_time'] = valid_time.strftime("%Y%m%dT%H%MZ")

        total_path_to_save = f"{output_dir}/{dt}_{accum}hr_accums.nc"
        lsr_path_to_save = f"{output_dir}/{dt}_{accum}hr_lsr_accums.nc"
        conv_path_to_save = f"{output_dir}/{dt}_{accum}hr_conv_accums.nc"
        # iris.save(lsr_accumulations, lsr_path_to_save)
        # iris.save(conv_accumulations, conv_path_to_save)
        iris.save(total_accumulations, total_path_to_save)
        # print(f"LARGE SCALE PRECIP: {lsr_accumulations}")
        # print("*********************************************")
        # print(f"CONVECTIVE PRECIP: {conv_accumulations}")
        print(f"TOTAL PRECIP: {total_accumulations}")
    
    # TRIAL OPTIONS
    # trial_file1 = f"/<path>/PS47/PS47_thresholdplot_data/{cube_dt}_{trial_name1}.pp"
    # trial_file2 = f"/<path>/PS47/PS47_thresholdplot_data/{cube_dt}_{trial_name2}.pp"


if __name__ == "__main__":
    main()
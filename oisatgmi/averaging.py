import numpy as np
import datetime
from oisatgmi.config import satellite_amf, satellite_opt

def averaging(startdate: str, enddate: str, reader_obj):
    """
    Average satellite and CTM data over a requested date range.

    The date interval is:
        startdate <= observation date < enddate

    Therefore:
        averaging('2023-01-01', '2023-01-16', ...)
            -> January 1-15

        averaging('2023-01-16', '2023-02-01', ...)
            -> January 16-31

    Inputs
    ------
    startdate : str
        Starting date in YYYY-mm-dd format.

    enddate : str
        Ending date in YYYY-mm-dd format. This date is exclusive.

    reader_obj :
        OI-SAT-GMI reader object containing satellite data.

    Returns
    -------
    sat_averaged_vcd
    sat_averaged_error
    ctm_averaged_vcd
    sat_aux1
    sat_aux2
    avg_datetime
    time_chosen
    mask_sat
    sat_tropopause
    """

    # ---------------------------------------------------------
    # Parse requested date range
    # ---------------------------------------------------------
    start_date = datetime.datetime.strptime(
        startdate, "%Y-%m-%d"
    ).date()

    end_date = datetime.datetime.strptime(
        enddate, "%Y-%m-%d"
    ).date()

    if end_date <= start_date:
        raise ValueError(
            f"enddate ({enddate}) must be later than "
            f"startdate ({startdate})"
        )

    # ---------------------------------------------------------
    # Find satellite granules inside requested period
    # ---------------------------------------------------------
    sat_chosen = []

    for sat_data in reader_obj.sat_data:

        if sat_data is None:
            continue

        time_sat = sat_data.time

        if isinstance(time_sat, datetime.datetime):
            time_sat_date = time_sat.date()
        elif isinstance(time_sat, datetime.date):
            time_sat_date = time_sat
        else:
            raise TypeError(
                "sat_data.time must be datetime.datetime "
                "or datetime.date"
            )

        if start_date <= time_sat_date < end_date:
            sat_chosen.append(sat_data)

    if len(sat_chosen) == 0:
        raise ValueError(
            f"No satellite observations found between "
            f"{startdate} and {enddate}"
        )

    print(
        f"Averaging {len(sat_chosen)} satellite granules "
        f"from {startdate} to {enddate} (end date exclusive)"
    )

    # ---------------------------------------------------------
    # Determine output grid
    # ---------------------------------------------------------
    grid_shape = np.shape(sat_chosen[0].vcd)

    if len(grid_shape) != 2:
        raise ValueError(
            "Expected satellite VCD to be a 2-D field, "
            f"but found shape {grid_shape}"
        )

    # ---------------------------------------------------------
    # Allocate running sums and counts
    #
    # This avoids creating large arrays such as:
    #
    #     np.array(sat_chosen_vcd)
    #
    # containing every granule simultaneously.
    # ---------------------------------------------------------

    sat_vcd_sum = np.zeros(grid_shape, dtype=np.float64)
    sat_vcd_count = np.zeros(grid_shape, dtype=np.int32)

    ctm_vcd_sum = np.zeros(grid_shape, dtype=np.float64)
    ctm_vcd_count = np.zeros(grid_shape, dtype=np.int32)

    tropopause_sum = np.zeros(grid_shape, dtype=np.float64)
    tropopause_count = np.zeros(grid_shape, dtype=np.int32)

    aux1_sum = np.zeros(grid_shape, dtype=np.float64)
    aux1_count = np.zeros(grid_shape, dtype=np.int32)

    aux2_sum = np.zeros(grid_shape, dtype=np.float64)
    aux2_count = np.zeros(grid_shape, dtype=np.int32)

    # For uncertainty of the mean:
    #
    # sigma_mean = sqrt(sum(sigma_i^2)) / N
    #
    error_variance_sum = np.zeros(grid_shape, dtype=np.float64)
    error_count = np.zeros(grid_shape, dtype=np.int32)

    time_chosen = []
    mask_sat = []

    # ---------------------------------------------------------
    # Small internal helper for accumulating averages
    # ---------------------------------------------------------
    def accumulate(field, running_sum, running_count,
                   required=False, field_name="field"):

        arr = np.asarray(field, dtype=np.float64)

        if arr.shape != grid_shape:

            if required:
                raise ValueError(
                    f"{field_name} has shape {arr.shape}, "
                    f"but expected {grid_shape}"
                )

            # Optional fields such as tropopause or auxiliary
            # variables may not exist for all products.
            return

        valid = np.isfinite(arr)

        running_sum[valid] += arr[valid]
        running_count[valid] += 1

    # ---------------------------------------------------------
    # Process each selected granule
    # ---------------------------------------------------------
    for sat_data in sat_chosen:

        time_chosen.append(sat_data.time)

        # -------------------------
        # Satellite VCD
        # -------------------------
        sat_vcd = np.asarray(
            sat_data.vcd,
            dtype=np.float64
        )

        if sat_vcd.shape != grid_shape:
            raise ValueError(
                f"Satellite VCD has inconsistent shape "
                f"{sat_vcd.shape}; expected {grid_shape}"
            )

        valid_sat = np.isfinite(sat_vcd)

        sat_vcd_sum[valid_sat] += sat_vcd[valid_sat]
        sat_vcd_count[valid_sat] += 1

        # Keep individual masks because driver.write_to_nc()
        # writes mask_satellite and frequency.
        mask_sat.append(valid_sat)

        # -------------------------
        # CTM VCD
        # -------------------------
        accumulate(
            sat_data.ctm_vcd,
            ctm_vcd_sum,
            ctm_vcd_count,
            required=True,
            field_name="ctm_vcd"
        )

        # -------------------------
        # Satellite uncertainty
        # -------------------------
        uncertainty = np.asarray(
            sat_data.uncertainty,
            dtype=np.float64
        )

        if uncertainty.shape != grid_shape:
            raise ValueError(
                f"Satellite uncertainty has shape "
                f"{uncertainty.shape}; expected {grid_shape}"
            )

        valid_error = np.isfinite(uncertainty)

        error_variance_sum[valid_error] += (
            uncertainty[valid_error] ** 2
        )

        error_count[valid_error] += 1

        # -------------------------
        # Tropopause
        # -------------------------
        accumulate(
            sat_data.tropopause,
            tropopause_sum,
            tropopause_count
        )

        # -------------------------
        # Auxiliary quantities
        # -------------------------
        if isinstance(sat_data, satellite_amf):

            accumulate(
                sat_data.new_amf,
                aux1_sum,
                aux1_count
            )

            accumulate(
                sat_data.old_amf,
                aux2_sum,
                aux2_count
            )

        elif isinstance(sat_data, satellite_opt):

            accumulate(
                sat_data.x_col,
                aux1_sum,
                aux1_count
            )

            accumulate(
                sat_data.ctm_xcol,
                aux2_sum,
                aux2_count
            )

    # ---------------------------------------------------------
    # Convert sums to averages
    # ---------------------------------------------------------
    def finalize_average(running_sum, running_count):

        output = np.full(
            grid_shape,
            np.nan,
            dtype=np.float64
        )

        valid = running_count > 0

        output[valid] = (
            running_sum[valid] /
            running_count[valid]
        )

        return output

    sat_averaged_vcd = finalize_average(
        sat_vcd_sum,
        sat_vcd_count
    )

    ctm_averaged_vcd = finalize_average(
        ctm_vcd_sum,
        ctm_vcd_count
    )

    sat_tropopause = finalize_average(
        tropopause_sum,
        tropopause_count
    )

    sat_aux1 = finalize_average(
        aux1_sum,
        aux1_count
    )

    sat_aux2 = finalize_average(
        aux2_sum,
        aux2_count
    )

    # ---------------------------------------------------------
    # Uncertainty of averaged satellite field
    #
    # Equivalent to the original:
    #
    # sqrt(sum(sigma_i^2) / N^2)
    # ---------------------------------------------------------
    sat_averaged_error = np.full(
        grid_shape,
        np.nan,
        dtype=np.float64
    )

    valid_error = error_count > 0

    sat_averaged_error[valid_error] = (
        np.sqrt(error_variance_sum[valid_error]) /
        error_count[valid_error]
    )

    # ---------------------------------------------------------
    # Mean observation time
    #
    # Avoid datetime.timestamp()/fromtimestamp() because those
    # can depend on the machine's local timezone.
    # ---------------------------------------------------------
    reference_time = time_chosen[0]

    time_offsets = np.array([
        (t - reference_time).total_seconds()
        for t in time_chosen
    ])

    avg_datetime = (
        reference_time +
        datetime.timedelta(
            seconds=float(np.mean(time_offsets))
        )
    )

    return (
        sat_averaged_vcd,
        sat_averaged_error,
        ctm_averaged_vcd,
        sat_aux1,
        sat_aux2,
        avg_datetime,
        time_chosen,
        mask_sat,
        sat_tropopause
    )

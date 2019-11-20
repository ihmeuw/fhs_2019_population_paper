"""This is a script for running aggregate_mortality_with_arima_error
for all fatal causes in the hierarchy. Full documentation of
the aggregation process is in aggregate_mortality_with_arima_error.py.
Aggregation occurs up the cause hierarchy, starting with modeled causes
and moving up to all cause mortality.
"""

import logging
import subprocess
import sys

try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path

from sqlalchemy.orm import sessionmaker

from fbd_core import db, argparse
from fbd_core.strategy_set.strategy import get_strategy_set
from fbd_core.strategy_set import get_hierarchy
from fbd_scenarios import aggregate_mortality_with_arima_error, subset_fatal

logger = logging.getLogger("run-aggregate-with-arima-error")

NTD_CAUSES = ("_ntd", "malaria", "ntd_chagas", "ntd_leish", "ntd_leish_visc",
              "ntd_leish_cut", "ntd_afrtryp", "ntd_schisto", "ntd_cysticer",
              "ntd_echino", "ntd_lf", "ntd_oncho", "ntd_trachoma", "ntd_dengue",
              "ntd_yellowfever", "ntd_rabies", "ntd_nema", "ntd_nema_ascar",
              "ntd_nema_trichur", "ntd_nema_hook", "ntd_foodborne", "ntd_other")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
            "--input-version", type=str, required=True,
            help="The mortality or yld version that the results are based on.")
    parser.add_argument(
            "--agg-version", type=str,
            help="The version the aggregations are saved under."
            )
    parser.add_argument(
            "--arima-version", type=str,
            help="The version the arima results are saved under."
            )
    parser.add_argument(
            "--measure", type=str, required=True,
            help="yld or death.")
    parser.add_argument(
            "--period", choices=["past", "future"], required=False,
            default="future",
            help="For aggs, whether to aggregate the past or the future")
    parser.add_argument(
            "--make-yhat", action="store_true",
            help="Make the mean mortality aggregates."
            )
    parser.add_argument(
            "--make-ystar", action="store_true",
            help="Make the sum of the means and the ARIMAed residual."
            )
    parser.add_argument(
            "--make-no-arima", action="store_true",
            help="Make a set of ystar results with no ARIMA"
            )
    parser.add_argument(
            "--wait-for-finish", action="store_true",
            help="Make the script not finish until the final cause is arima'd."
                 " Used by misfire."
            )
    parser.add_argument(
            "--intercept-shift", action="store_true",
            help="Shift the draws to account for past uncertainty."
            )
    parser.add_argument(
            "--gbd-round-id", type=int, required=True,
            help="the gbd round estimates are based off of,"
            "e.g., 3 for 2015")
    parser.add_argument(
            "--past-version", help="past version to pull results from"
    )
    parser.set_defaults(past_version=VERSION)
    parser.add_arg_years()
    parser.add_arg_draws()
    parser.add_arg_dryrun()

    args = parser.parse_args()

    memory = 170 if args.draws == 1000 else 55
    threads = 20
    years = str(args.years)

    python = sys.executable
    script = (Path(aggregate_mortality_with_arima_error.__file__)
              .absolute()
              .as_posix()
              )
    qsub_tmp = (
        "qsub -N v{{input_version}}-{{acause}} -hold_jid {{jids}} -now no "
        "-l m_mem_free={{memory}}G -l fthread={{threads}} -q all.q -b y -P proj_forecasting -terse "
        "{python} {script} "
        "--input-version {{input_version}} --agg-version {{agg_version}} "
        "--arima-version {{arima_version}} --measure {{measure}} "
        "--period {{period}} --acause {{acause}} --smoothing {{smoothing}} "
        "--years {{years}} --draws {{draws}} --make-{{make}} "
        "{{intercept_shift}} {{no_arima}} -v --gbd-round-id {{gbd_round_id}} "
        "--past-version {{past_version}}"
        ).format(python=python, script=script)
    dryrun_qsub_tmp = (
            "qsub -terse -N TEST-{acause} -hold_jid {jids} -b y sleep 3")

    # Mapping from cause level to the dimensions to smooth over for the ARIMA.
    smoothing = {
       0: ["location_id", "sex_id", "age_group_id"],
        1: ["location_id", "sex_id", "age_group_id"],
        2: ["region_id", "sex_id", "age_group_id"],
        3: ["super_region_id", "sex_id", "age_group_id"],
        "modeled": ["super_region_id", "sex_id", "age_group_id"]
        }

    # create a session to read in the relevant causes
    engine = db.db_engine(NAME, DATABASE)
    session = sessionmaker(bind=engine)()
    all_causes = get_hierarchy(session, "cause", CAUSE_HIERARCHY_ID)[
            ["acause", "cause_id", "parent_id", "level"]]

    cause_strategy_set = get_strategy_set(
                session, GK_STRATEGY_SET_ID, CAUSE_HIERARCHY_ID)
    cause_hierarchy = get_hierarchy(session, "cause", CAUSE_HIERARCHY_ID)
    cause_tree, node_map = subset_fatal.make_hierarchy_tree(cause_hierarchy, 294, "cause_id")
    fatal_subset = subset_fatal.include_up_hierarchy(cause_tree, node_map,
                                                     cause_strategy_set["cause_id"].values)
    fatal_causes = all_causes[all_causes.cause_id.isin(fatal_subset)]
    modeled_causes = fatal_causes.query("cause_id not in parent_id.values")
    aggregate_causes = fatal_causes.query("cause_id in parent_id.values")

    # Grab causes that are not modeled. There should be no level 4 causes.
    assert len(aggregate_causes.query("level == 4")) == 0

    hold_jids = ["1"]
    arima_jids = ["1"]
    for level in ["modeled", 3, 2, 1, 0]:
        current_jids = ["1"]

        if level == "modeled":
            current_causes = modeled_causes
        else:
            current_causes = aggregate_causes.query("level == @level")

        if args.make_yhat:
            logger.info("Submitting yhat jobs for level %s" % str(level))
            for acause in current_causes["acause"].values:
                qsub = qsub_tmp.format(
                    jids=",".join(hold_jids),
                    memory=memory,
                    threads=threads,
                    input_version=args.input_version,
                    agg_version=args.agg_version,
                    arima_version=args.arima_version,
                    measure=args.measure, period=args.period,
                    smoothing=" ".join(smoothing[level]),
                    acause=acause, years=years, draws=args.draws,
                    make="yhat", intercept_shift="",
                    no_arima="",
                    gbd_round_id=args.gbd_round_id,
                    past_version=args.past_version
                    )
                logger.debug(qsub)
                if args.dryrun:
                    qsub = dryrun_qsub_tmp.format(
                        acause=acause,
                        jids=",".join(hold_jids))
                    logger.debug(qsub)
                output = subprocess.check_output(qsub, shell=True)
                current_jids.append(output[:-1].decode("utf-8"))  # ignore newline.

        hold_jids = current_jids
        intercept_shift = "--intercept-shift" if args.intercept_shift else ""

        if args.make_no_arima:
            logger.info("Submitting no-arima jobs for level %s" % str(level))
            for acause in current_causes["acause"].values:
                qsub = qsub_tmp.format(
                    jids=",".join(hold_jids),
                    memory=memory,
                    threads=threads,
                    input_version=args.input_version,
                    agg_version=args.agg_version,
                    arima_version=args.arima_version + "_no_arima",
                    measure=args.measure, period=args.period,
                    smoothing=" ".join(smoothing[level]),
                    acause=acause, years=years, draws=args.draws,
                    make="ystar", intercept_shift=intercept_shift,
                    no_arima="--no-arima",
                    gbd_round_id=args.gbd_round_id,
                    past_version=args.past_version
                )
                logger.debug(qsub)
                if args.dryrun:
                    qsub = dryrun_qsub_tmp.format(
                        acause=acause,
                        jids=",".join(hold_jids)
                    )
                    logger.debug(qsub)
                output = subprocess.check_output(qsub, shell=True)
        if args.make_ystar:
            logger.info("Submitting ystar jobs for level %s" % str(level))
            for acause in current_causes["acause"].values:
                no_arima = "--no-arima" if acause in NTD_CAUSES else ""
                qsub = qsub_tmp.format(
                    jids=",".join(hold_jids),
                    memory=memory,
                    threads=threads,
                    input_version=args.input_version,
                    agg_version=args.agg_version,
                    arima_version=args.arima_version,
                    measure=args.measure, period=args.period,
                    smoothing=" ".join(smoothing[level]),
                    acause=acause, years=years, draws=args.draws,
                    make="ystar", intercept_shift=intercept_shift,
                    no_arima=no_arima,
                    gbd_round_id = args.gbd_round_id,
                    past_version = args.past_version
                )
                logger.debug(qsub)
                if args.dryrun:
                    qsub = dryrun_qsub_tmp.format(
                        acause=acause,
                        jids=",".join(hold_jids))

                output = subprocess.check_output(qsub, shell=True)
                arima_jids.append(output[:-1])  # ignore newline.

    # Sync on a trivial followup job that wont run until all the squeezes do
    if args.wait_for_finish:
        trivial_qsub = "qsub -N endsqueeze -hold_jid {jids:s} -sync y ls"
        trivial_qsub = trivial_qsub.format(jids=','.join(arima_jids))

        output = subprocess.check(trivial_qsub)




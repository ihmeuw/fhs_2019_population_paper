"""This is a script for running squeeze.py for all of the causes in the
heirarchy. Full documentation of the squeeze process is in squeeze.py.

The squeeze begins with the _all cause, and proceeds down to modeled causes.
"""

import logging
import subprocess
import sys

try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path

import pandas as pd
from fbd_core import db, argparse

from fbd_scenarios import squeeze
from fbd_scenarios.subset_fatal import get_fatal_causes

logger = logging.getLogger("run-squeeze")

EXTERNAL_CAUSES = pd.read_csv(_EXTERNAL_CAUSES_CSV)['acause'].values.tolist()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
            "--star-version", type=str,
            help=("The version the results with ARIMAed residuals are saved "
                  "under.")
            )
    parser.add_argument(
            "--squeeze-version", type=str, required=True,
            help="The mortality or yld version for which results are saved.")
    parser.add_argument(
            "--measure", type=str, required=True,
            help="death or yld.")
    parser.add_argument(
            "--wait-for-finish", action="store_true",
            help="Make the script not finish until the final cause is arima'd."
                 " Used by misfire."
            )
    parser.add_argument(
            "--make-lex", action="store_true",
            help="Whether to calculate life expectancy from all-cause "
            "mortality and save it to a netcdf in the squeeze directory")
    parser.add_argument(
            "--gbd-round-id", type=int, required=True,
            help="The gbd round to base estimates off of,"
            "e.g., 3 for 2015")

    parser.add_arg_years()
    parser.add_arg_dryrun()
    parser.add_arg_draws()

    args = parser.parse_args()

    script = (Path(squeeze.__file__)
              .absolute()
              .as_posix()
              )

    make_lex = "--make-lex" if args.make_lex else ""
    memory = 200 if args.draws == 1000 else 60

    qsub_tmp = (
        "qsub -N n{squeeze_version}-{{parent_cause}} -hold_jid {{jids}} "
        "-l m_mem_free={memory}G -l fthread=20 -q all.q "
        "-now no -b y -P proj_forecasting -terse -V "
        "{python} {script} --star-version {star_version} --squeeze-version "
        "{squeeze_version} --measure {measure} {make_lex} "
        "--parent-cause {{parent_cause}} --gbd-round-id {gbd_round_id} "
        "--years {years}"
        ).format(memory=memory, python=python, script=script, star_version=args.star_version,
                 squeeze_version=args.squeeze_version, measure=args.measure,
                 make_lex=make_lex,
                 gbd_round_id=args.gbd_round_id,
                 years=args.years)
    dryrun_qsub_tmp = (
        "qsub -N TEST_{parent_cause} -hold_jid {jids} -b y sleep 3")

    if args.measure == "yld":
        cause_df = db.get_cause_set("nonfatal", fbd_round_id=args.fbd_round_id)
    else:
        fatal_causes = get_fatal_causes(args.gbd_round_id)[["acause",
                                                            "cause_id",
                                                            "parent_id",
                                                            "level"]]
    causes = fatal_causes.query("cause_id in parent_id.values")

    hold_jids = ["1"]
    for level in [0, 1, 2, 3]:
        current_jids = ["1"]
        current_causes = causes.query("level == @level")
        for acause in current_causes["acause"].values:
            if acause in EXTERNAL_CAUSES:
                logger.info(
                    "{} is an external cause. Not squeezing.".format(acause))
                continue
            elif acause == "_shock":
                logger.info(
                    "shock is an aggregate of external causes. Not squeezing.")
                continue
            else:
                logger.info("{} will be squeezed".format(acause))

            qsub = qsub_tmp.format(
                jids=",".join(hold_jids), parent_cause=acause)
            logger.debug(qsub)
            if args.dryrun:
                qsub = dryrun_qsub_tmp.format(
                    jids=",".join(hold_jids), parent_cause=acause)
                logger.debug(qsub)
            output = subprocess.check_output(qsub, shell=True).decode()
            current_jids.append(output[:-1])  # ignore newline character.

        hold_jids = current_jids

    # Sync on a trivial followup job that wont run until all the squeezes do
    if args.wait_for_finish:
        trivial_qsub = "qsub -N endsqueeze -hold_jid {jids:s} -sync y ls"
        trivial_qsub = trivial_qsub.format(jids=','.join(current_jids))

        output = subprocess.check(trivial_qsub)

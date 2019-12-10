#!/usr/bin/env python
import os
from shutil import copyfile
from fbd_core import argparse, db
from fbd_core.file_interface import FBDPath
from fbd_core.strategy_set.strategy import get_strategy_set
from fbd_cod import settings
from sqlalchemy.orm import sessionmaker


def make_run_log_file(version):
    """
    Right now just copies the settings file to the output directory so
    people can see which versions and whatnot were used.

    Args:
        version (str): version name where the current mortality run is to be
        saved
    """
    # make the run directory so this log can be saved
    run_dir = FBDPath("/{gri}/future/death/{v}/".format(
            gri=settings.GBD_ROUND_ID, v=version))
    run_dir.mkdir(exist_ok=True)

    # get the source and destination paths
    source_path = os.path.join(os.pardir, "fbd_cod/settings.py")
    dest_path = os.path.join(str(run_dir), "versions.py")

    copyfile(source_path, dest_path)


def main(version, years, draws, oos, subnational, dryrun=False):
    """
    Run all cause-specific cod models with the current modeling framework.
    This file must be run from the `scripts` directory of fbd_cod.

    Args:
        model (str): version name to use for the current mortality run.
        years (fbd_core.argparse.YearRange): years to load and model, e.g.
            (1990:2017:2040).
        draws (int): how many draws to run through the pipeline
        oos (bool): whether to hold a time series out of sample for validation
            and comparison
        subnational (bool): whether or not to include the 93 subnational csu
            locations in the model
    """
    # NOTE: This wont work if you install fbd_cod and attempt to run
    # this script from the command line, but it never did
    file_dir = os.path.dirname(os.path.realpath(__file__))
    execf_ = os.path.join(file_dir, "run_cod_model.py")

    make_run_log_file(version)
    years_str = years.__str__()
    threads = 30

    qsub_template = (
            "qsub -b y -l m_mem_free={memory}G -l fthread={threads} -q all.q -now no -P proj_forecasting "
            "-N {acause}_{sex}_{version}{dryrun} "
            "{python} {exec_file} -v "
            "-c {acause} -s {sex} --version {version} --years {years} --draws {"
            "draws} "
            #"--spline {sdi_interaction} {oos} {subnat} {dryrun}")
            "{sdi_interaction} {oos} {subnat} {dryrun}")

    # create a db connection to get the strategy set for fatal GK causes
    engine = db.db_engine("fbd-dev-read", database="forecasting")
    session = sessionmaker(bind=engine)()
    cols_to_keep = ["acause", "male", "female"]
    fatal_gk_causes = get_strategy_set(session, 18, 303)[cols_to_keep]

    for _, row in fatal_gk_causes.iterrows():
        for sex_id in settings.SEX_DICT.keys():
            sex_name = settings.SEX_DICT[sex_id]
            if not row[sex_name] == 1:
                continue
            acause = row["acause"]
            if acause in settings.INTERACTION_CAUSES:
                sdi_interaction = "--sdi-interaction"
            else:
                sdi_interaction = ""
            oos_arg = "--oos" if oos else ""
            subnat_arg = "--subnational" if subnational else ""
            dryrun_arg = "--dryrun" if dryrun else ""
            if (acause.startswith("ckd")) or (acause == "nutrition_pem"):
                memory = 500 if draws == 1000 else 75
            else:
                memory = 400 if draws == 1000 else 75

            qsub = qsub_template.format(
                memory=memory,
                threads=threads,
                acause=acause,
                sex=sex_id,
                years=years_str,
                version=version,
                exec_file=execf_,
                draws=draws,
                python=settings.PYTHON_EXEC,
                sdi_interaction=sdi_interaction,
                oos=oos_arg,
                subnat=subnat_arg,
                dryrun=dryrun_arg)
            print(qsub)
            os.popen(qsub)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description='Run last year constant model')
    parser.add_arg_years()
    parser.add_arg_draws()
    parser.add_arg_dryrun()
    parser.add_argument('--version', type=str, required=True,
                        help='run name/version id')
    parser.add_argument('--oos', action='store_true',
                        help='Whether to run with out-of-sample inputs')
    parser.add_argument('--subnational', action='store_true',
                        help=('Whether or not to run including the 93 '
                              'subnational locations used for CSU'))
    args = parser.parse_args()
    main(args.version, args.years, args.draws, args.oos, args.subnational,
         args.dryrun)

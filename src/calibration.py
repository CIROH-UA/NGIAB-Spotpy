from spotpy.parameter import Uniform
import cli


CFE_PARAMS = {
    "b": Uniform(2.0, 15.0, optguess=4.05),
    "satpsi": Uniform(0.03, 0.955, optguess=0.355),
    "satdk": Uniform(1e-7, 7.26e-4, optguess=3.38e-6),
    "maxsmc": Uniform(0.16, 0.59, optguess=0.439),
    "refkdt": Uniform(0.1, 4.0, optguess=1.0),
    "expon": Uniform(1.0, 8.0, optguess=3.0),
    "slope": Uniform(0.0, 1.0, optguess=0.1),
    "max_gw_storage": Uniform(0.01, 0.25, optguess=0.05),
    "Kn": Uniform(0.0, 1.0, optguess=0.03),
    "Klf": Uniform(0.0, 1.0, optguess=0.01),
    "Cgw": Uniform(1.8e-6, 1.8e-3, optguess=1.8e-5),
}

NOAH_PARAMS = {
    "MFSNO": Uniform(0.5, 4.0, optguess=2.0),
    "MP": Uniform(3.6, 12.6, optguess=9.0),
    "RSURF_EXP": Uniform(1.0, 6.0, optguess=5.0),
    "CWP": Uniform(0.09, 0.36, optguess=0.18),
    "VCMX25": Uniform(24.0, 112.0, optguess=52.2),
    "RSURF_SNOW": Uniform(0.136, 100.0, optguess=50.0),
    "SCAMAX": Uniform(0.7, 1.0, optguess=0.9),
}

# Keys must match the `model_type_name` values in `realization.json`.
CALIBRATION_PARAMS = {"CFE": CFE_PARAMS, "NoahOWP": NOAH_PARAMS}


def main() -> int:
    cli.set_calibration_params(CALIBRATION_PARAMS)
    return int(cli.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())

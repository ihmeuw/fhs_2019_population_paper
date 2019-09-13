##### run out to 2100, no oos yet

### THINGS YOU WILL LIKELY BE CHANGING A LOT

# first entry is version, second entry is gbd_round_id
VERSIONS = dict(
    past_mortality=(VERSION, 5),
    sdi=(VERSION, 5),
    scalar=(VERSION, 5),
    sev=(VERSION, 5),
    asfr=(VERSION, 5),
    past_asfr = (VERSION, 5),
    hiv=(VERSION, 5),
    vehicles_2_plus_4wheels_pc=(VERSION, 5)
)


### THINGS YOU WILL LIKELY NOT BE CHANGING VERY OFTEN

FLOOR = 1e-28
SDI_KNOT = 0.8
SCENARIOS = [-1, 0, 1]
SCALAR_CAP = 50.
SEX_DICT = {1: "male", 2: "female"}

# causes to drop SDI for because it's already in the vaccine model
VACCINE_CAUSES = ("tetanus", "diptheria", "whooping", "measles",
                  "diarrhea_rotavirus", "meningitis_hib", "otitis",
                  "meningitis_pneumo")

# causes to include an SDI*time interaction for because of epi transition
INTERACTION_CAUSES = ("cvd_stroke_cerhem", "cvd_stroke_isch", "cvd_ihd",
                      "cvd_htn", "cirrhosis_alcohol", "cirrhosis_hepb",
                      "cirrhosis_hepc", "cirrhosis_other", "diabetes",
                      "ckd_diabetes", "ckd_glomerulo", "ckd_htn", "ckd_other")


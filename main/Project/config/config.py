from decimal import Decimal

RAW_REPORT_HISTORY_TRAIN_FILE = "data/report_history_train.csv"
RAW_REPORT_HISTORY_EVAL_FILE = "data/report_history_eval.csv"
RAW_REPAIR_LOGS_TRAIN_FILE = "data/repair_logs_train.csv"
RAW_REPAIR_LOGS_EVAL_FILE = "data/repair_logs_eval.csv"
RAW_STATEMENT_SENTENCES_FILE = "data/statement_sentences.csv"

CLEAN_REPORT_HISTORY_TRAIN_FILE = "processed/report_history_train_clean.csv"
CLEAN_REPORT_HISTORY_EVAL_FILE = "processed/report_history_eval_clean.csv"
CLEAN_REPAIR_LOGS_TRAIN_FILE = "processed/repair_logs_train_clean.csv"
CLEAN_REPAIR_LOGS_EVAL_FILE = "processed/repair_logs_eval_clean.csv"
CLEAN_STATEMENT_SENTENCES_FILE = "processed/statement_sentences_clean.csv"

DATA_QUALITY_REPORT_FILE = "output/data_quality_report.json"
JSON_INDENT = 2
TARGET_SHARE_DECIMALS = 4
DATASET_NAME_WIDTH = 30
ROW_COUNT_WIDTH = 6


NULL_TOKENS = ["", "n/a", "na", "-", "null", "not known", "caller unsure"]


WEBSITE = "Website"
STREETCARE_APP = "StreetCare app"
CONTACT_CENTRE = "Contact centre"

REPORT_CHANNEL_MAPPING = {
    "website": WEBSITE,
    "streetcare app": STREETCARE_APP,
    "contact centre": CONTACT_CENTRE,
}


BOOLEAN_YES = "Yes"
BOOLEAN_NO = "No"

BOOLEAN_MAPPING = {
    "yes": BOOLEAN_YES,
    "y": BOOLEAN_YES,
    "true": BOOLEAN_YES,
    "1": BOOLEAN_YES,
    "no": BOOLEAN_NO,
    "n": BOOLEAN_NO,
    "false": BOOLEAN_NO,
    "0": BOOLEAN_NO,
}


RESIDENT = "Resident"
BUSINESS = "Business"
COUNCILLOR = "Councillor"
BUS_OPERATOR = "Bus operator"
SCHOOL = "School"

REPORTER_TYPE_MAPPING = {
    "resident": RESIDENT,
    "business": BUSINESS,
    "councillor": COUNCILLOR,
    "bus operator": BUS_OPERATOR,
    "school": SCHOOL,
}


TOWN_CENTRE = "Town Centre"
NORTH = "North"
SOUTH = "South"
VILLAGES = "Villages"

DISTRICT_MAPPING = {
    "town centre": TOWN_CENTRE,
    "north": NORTH,
    "south": SOUTH,
    "villages": VILLAGES,
}

STREET_NAME_MAPPING = {
    "high street": "High Street",
    "market place": "Market Place",
    "church street": "Church Street",
    "station road": "Station Road",
    "bridge street": "Bridge Street",
    "king street": "King Street",
    "castle street": "Castle Street",
    "mill lane": "Mill Lane",
    "chapel row": "Chapel Row",
    "bank parade": "Bank Parade",
    "sheep street": "Sheep Street",
    "wool lane": "Wool Lane",
    "northgate": "Northgate",
    "park avenue": "Park Avenue",
    "hillside road": "Hillside Road",
    "beech grove": "Beech Grove",
    "oakfield drive": "Oakfield Drive",
    "meadow close": "Meadow Close",
    "cedar way": "Cedar Way",
    "kingsley road": "Kingsley Road",
    "orchard rise": "Orchard Rise",
    "windmill lane": "Windmill Lane",
    "grange road": "Grange Road",
    "hawthorn close": "Hawthorn Close",
    "london road": "London Road",
    "southfield road": "Southfield Road",
    "riverside walk": "Riverside Walk",
    "victoria road": "Victoria Road",
    "albert street": "Albert Street",
    "canal street": "Canal Street",
    "elm terrace": "Elm Terrace",
    "willow crescent": "Willow Crescent",
    "brook lane": "Brook Lane",
    "foundry lane": "Foundry Lane",
    "tannery close": "Tannery Close",
    "ferry lane": "Ferry Lane",
    "ashby road": "Ashby Road",
    "long lane": "Long Lane",
    "back lane": "Back Lane",
    "school lane": "School Lane",
    "the green": "The Green",
    "pond lane": "Pond Lane",
    "holloway lane": "Holloway Lane",
    "top road": "Top Road",
    "moor lane": "Moor Lane",
    "clay pit lane": "Clay Pit Lane",
    "littleworth": "Littleworth",
    "kiln lane": "Kiln Lane",
}


TRAFFIC_SENSITIVE = "Traffic sensitive"
LOCAL_ROAD = "Local road"

STREET_CATEGORY_MAPPING = {
    "traffic sensitive": TRAFFIC_SENSITIVE,
    "main route": TRAFFIC_SENSITIVE,
    "ts": TRAFFIC_SENSITIVE,
    "local road": LOCAL_ROAD,
    "residential street": LOCAL_ROAD,
    "lr": LOCAL_ROAD,
}


CARRIAGEWAY = "Carriageway"
FOOTWAY = "Footway"
DRAINAGE = "Drainage"
VEGETATION = "Vegetation"
STREET_LIGHTING = "Street lighting"

DEFECT_TYPE_MAPPING = {
    "carriageway": CARRIAGEWAY,
    "pothole or road surface": CARRIAGEWAY,
    "cw": CARRIAGEWAY,
    "footway": FOOTWAY,
    "pavement defect": FOOTWAY,
    "fw": FOOTWAY,
    "drainage": DRAINAGE,
    "flooding or blocked drain": DRAINAGE,
    "dr": DRAINAGE,
    "vegetation": VEGETATION,
    "overhanging tree or hedge": VEGETATION,
    "vg": VEGETATION,
    "street lighting": STREET_LIGHTING,
    "street light fault": STREET_LIGHTING,
    "sl": STREET_LIGHTING,
}

NO_LINKED_DEFECT = "No linked defect"

LINKED_DEFECT_TYPE_MAPPING = {
    **DEFECT_TYPE_MAPPING,
    "no linked defect": NO_LINKED_DEFECT,
    "none": NO_LINKED_DEFECT,
    "nil": NO_LINKED_DEFECT,
}


CARRIAGEWAY_PART = "Carriageway"
FOOTWAY_PART = "Footway"

HIGHWAY_PART_MAPPING = {
    "carriageway": CARRIAGEWAY_PART,
    "road": CARRIAGEWAY_PART,
    "footway": FOOTWAY_PART,
    "pavement": FOOTWAY_PART,
    "path": FOOTWAY_PART,
}


LAMP_OUT = "Lamp out"
DAY_BURNING = "Day burning"
LEANING_COLUMN = "Leaning column"

LIGHTING_FAULT_MAPPING = {
    "lamp out": LAMP_OUT,
    "light not working": LAMP_OUT,
    "out": LAMP_OUT,
    "day burning": DAY_BURNING,
    "light on in daytime": DAY_BURNING,
    "day": DAY_BURNING,
    "leaning column": LEANING_COLUMN,
    "column leaning": LEANING_COLUMN,
    "lean": LEANING_COLUMN,
}


REPAIRED_AS_REPORTED = "Repaired as reported"
REPAIRED_AFTER_INSPECTION = "Repaired after inspection"
REPAIRED_AFTER_COORDINATION = "Repaired after coordination"
CLOSED_AFTER_INSPECTION = "Closed after inspection"

REVIEW_OUTCOME_MAPPING = {
    "repaired as reported": REPAIRED_AS_REPORTED,
    "repaired": REPAIRED_AS_REPORTED,
    "rep": REPAIRED_AS_REPORTED,
    "repaired after inspection": REPAIRED_AFTER_INSPECTION,
    "repaired - inspected": REPAIRED_AFTER_INSPECTION,
    "rep-ins": REPAIRED_AFTER_INSPECTION,
    "repaired after coordination": REPAIRED_AFTER_COORDINATION,
    "repaired - coordinated": REPAIRED_AFTER_COORDINATION,
    "rep-crd": REPAIRED_AFTER_COORDINATION,
    "closed after inspection": CLOSED_AFTER_INSPECTION,
    "closed - inspected": CLOSED_AFTER_INSPECTION,
    "cls-ins": CLOSED_AFTER_INSPECTION,
}


REPAIR_FAILED = "Repair failed"
DEFECT_RETURNED_NEARBY = "Defect returned nearby"
REGROWTH = "Regrowth"
BLOCKED_AGAIN = "Blocked again"
LAMP_FAILED_AGAIN = "Lamp failed again"
WORSENED_AFTER_INSPECTION = "Worsened after inspection"

REPEAT_CAUSE_MAPPING = {
    "repair failed": REPAIR_FAILED,
    "failed": REPAIR_FAILED,
    "defect returned nearby": DEFECT_RETURNED_NEARBY,
    "new defect nearby": DEFECT_RETURNED_NEARBY,
    "nearby": DEFECT_RETURNED_NEARBY,
    "regrowth": REGROWTH,
    "grown back": REGROWTH,
    "blocked again": BLOCKED_AGAIN,
    "reblocked": BLOCKED_AGAIN,
    "lamp failed again": LAMP_FAILED_AGAIN,
    "failed again": LAMP_FAILED_AGAIN,
    "refail": LAMP_FAILED_AGAIN,
    "worsened after inspection": WORSENED_AFTER_INSPECTION,
    "got worse": WORSENED_AFTER_INSPECTION,
    "worse": WORSENED_AFTER_INSPECTION,
}


PATCH_REPAIR = "Patch repair"
PERMANENT_REINSTATEMENT = "Permanent reinstatement"
SLAB_RELAY = "Slab relay"
GULLY_CLEANSE = "Gully cleanse"
DRAIN_REPAIR = "Drain repair"
CUT_BACK = "Cut back"
LAMP_REPLACEMENT = "Lamp replacement"
PHOTOCELL_REPLACEMENT = "Photocell replacement"
COLUMN_REPLACEMENT = "Column replacement"

REPAIR_METHOD_MAPPING = {
    "patch repair": PATCH_REPAIR,
    "patch": PATCH_REPAIR,
    "permanent reinstatement": PERMANENT_REINSTATEMENT,
    "permanent": PERMANENT_REINSTATEMENT,
    "perm": PERMANENT_REINSTATEMENT,
    "slab relay": SLAB_RELAY,
    "relay slabs": SLAB_RELAY,
    "relay": SLAB_RELAY,
    "gully cleanse": GULLY_CLEANSE,
    "cleanse gully": GULLY_CLEANSE,
    "gully": GULLY_CLEANSE,
    "drain repair": DRAIN_REPAIR,
    "repair drain": DRAIN_REPAIR,
    "drain": DRAIN_REPAIR,
    "cut back": CUT_BACK,
    "cut": CUT_BACK,
    "lamp replacement": LAMP_REPLACEMENT,
    "replace lamp": LAMP_REPLACEMENT,
    "lamp": LAMP_REPLACEMENT,
    "photocell replacement": PHOTOCELL_REPLACEMENT,
    "replace photocell": PHOTOCELL_REPLACEMENT,
    "pecu": PHOTOCELL_REPLACEMENT,
    "column replacement": COLUMN_REPLACEMENT,
    "replace column": COLUMN_REPLACEMENT,
    "column": COLUMN_REPLACEMENT,
}


NO_TRAFFIC_MANAGEMENT = "None"
SIGNS_ONLY = "Signs only"
LANE_CLOSURE = "Lane closure"
ROAD_CLOSURE = "Road closure"

TRAFFIC_MANAGEMENT_MAPPING = {
    "none": NO_TRAFFIC_MANAGEMENT,
    "no traffic management": NO_TRAFFIC_MANAGEMENT,
    "signs only": SIGNS_ONLY,
    "signs": SIGNS_ONLY,
    "lane closure": LANE_CLOSURE,
    "lane closed": LANE_CLOSURE,
    "lane": LANE_CLOSURE,
    "road closure": ROAD_CLOSURE,
    "road closed": ROAD_CLOSURE,
    "closed": ROAD_CLOSURE,
}


COMPLETED = "Completed"
COMPLETED_WITH_OVERRUN = "Completed with overrun"
COMPLETED_AFTER_RETURN_VISIT = "Completed after return visit"

COMPLETION_CODE_MAPPING = {
    "completed": COMPLETED,
    "done": COMPLETED,
    "comp": COMPLETED,
    "completed with overrun": COMPLETED_WITH_OVERRUN,
    "done - overran": COMPLETED_WITH_OVERRUN,
    "comp-ovr": COMPLETED_WITH_OVERRUN,
    "completed after return visit": COMPLETED_AFTER_RETURN_VISIT,
    "done - second visit": COMPLETED_AFTER_RETURN_VISIT,
    "comp-rtn": COMPLETED_AFTER_RETURN_VISIT,
}


RESIDENT_DESCRIPTION = "Resident description"
CALL_NOTE = "Call note"
PHOTO_NOTE = "Photo note"
INSPECTION_NOTE = "Inspection note"

DOCUMENT_TYPE_MAPPING = {
    "resident description": RESIDENT_DESCRIPTION,
    "call note": CALL_NOTE,
    "photo note": PHOTO_NOTE,
    "inspection note": INSPECTION_NOTE,
}


MEASUREMENT = "Measurement"
LOCATION = "Location"
PHOTOGRAPH = "Photograph"
HAZARD = "Hazard"
INCIDENT = "Incident"
TRAFFIC_AND_USE = "Traffic and use"
PREVIOUS_REPORTS = "Previous reports"
WEATHER_AND_WATER = "Weather and water"
BACKGROUND = "Background"

SENTENCE_TYPE_MAPPING = {
    "measurement": MEASUREMENT,
    "measurements": MEASUREMENT,
    "location": LOCATION,
    "where": LOCATION,
    "photograph": PHOTOGRAPH,
    "photo": PHOTOGRAPH,
    "hazard": HAZARD,
    "danger": HAZARD,
    "incident": INCIDENT,
    "accident": INCIDENT,
    "traffic and use": TRAFFIC_AND_USE,
    "traffic & use": TRAFFIC_AND_USE,
    "traffic": TRAFFIC_AND_USE,
    "traffic-and-use": TRAFFIC_AND_USE,
    "previous reports": PREVIOUS_REPORTS,
    "previous report": PREVIOUS_REPORTS,
    "previous-reports": PREVIOUS_REPORTS,
    "weather and water": WEATHER_AND_WATER,
    "weather & water": WEATHER_AND_WATER,
    "weather": WEATHER_AND_WATER,
    "weather-and-water": WEATHER_AND_WATER,
    "background": BACKGROUND,
    "context": BACKGROUND,
    "general": BACKGROUND,
}


REPORT_REFERENCE_PATTERN = r"^(?:SC(?=[0-9]{7}$)|SC-)?([0-9]{1,7})$"
REPORT_REFERENCE_PREFIX = "SC"
REPORT_REFERENCE_DIGITS = 7
CANONICAL_REPORT_REFERENCE_PATTERN = r"^SC[0-9]{7}$"
SENTENCE_ID_PATTERN = r"^SEN-[0-9]{7}$"

WHOLE_NUMBER_PATTERN = r"^([0-9]+)$"
POSITIVE_WHOLE_NUMBER_PATTERN = r"^([1-9][0-9]*)$"
SIGNED_WHOLE_NUMBER_PATTERN = r"^(-?[0-9]+)$"
WHOLE_NUMBER_DTYPE = "Int64"
DECIMAL_DTYPE = "float64"
TEXT_DTYPE = "str"

WHOLE_STEP = Decimal("1")
TENTH_STEP = Decimal("0.1")
NO_CONVERSION = Decimal("1")
MILLIMETRES_PER_CENTIMETRE = Decimal("10")
MILLIMETRES_PER_INCH = Decimal("25.4")
METRES_PER_FOOT = Decimal("0.3048")
MINUTES_PER_HOUR = Decimal("60")

MILLIMETRE_READING = {
    "notations": {"millimetres": r"(?i)^(-?[0-9]+(?:\.[0-9]+)?)$",
                  "centimetres": r"(?i)^(-?[0-9]+(?:\.[0-9]+)?) cm$",
                  "inches": r"(?i)^(-?[0-9]+(?:\.[0-9]+)?) in$"},
    "multipliers": {"millimetres": NO_CONVERSION, "centimetres": MILLIMETRES_PER_CENTIMETRE,
                    "inches": MILLIMETRES_PER_INCH},
    "divisors": {"millimetres": NO_CONVERSION, "centimetres": NO_CONVERSION,
                 "inches": NO_CONVERSION},
    "step": WHOLE_STEP,
    "dtype": WHOLE_NUMBER_DTYPE,
}

METRE_READING = {
    "notations": {"metres": r"(?i)^(-?[0-9]+(?:\.[0-9]+)?)(?: m)?$",
                  "feet": r"(?i)^(-?[0-9]+(?:\.[0-9]+)?) ft$"},
    "multipliers": {"metres": NO_CONVERSION, "feet": METRES_PER_FOOT},
    "divisors": {"metres": NO_CONVERSION, "feet": NO_CONVERSION},
    "step": TENTH_STEP,
    "dtype": DECIMAL_DTYPE,
}

DEGREE_READING = {
    "notations": {"degrees": r"(?i)^(-?[0-9]+(?:\.[0-9]+)?)(?: deg)?$"},
    "multipliers": {"degrees": NO_CONVERSION},
    "divisors": {"degrees": NO_CONVERSION},
    "step": TENTH_STEP,
    "dtype": DECIMAL_DTYPE,
}

CREW_HOURS_READING = {
    "notations": {"hours": r"(?i)^(-?[0-9]+(?:\.[0-9]+)?)(?: hrs?)?$",
                  "minutes": r"(?i)^(-?[0-9]+(?:\.[0-9]+)?) min$"},
    "multipliers": {"hours": NO_CONVERSION, "minutes": NO_CONVERSION},
    "divisors": {"hours": NO_CONVERSION, "minutes": MINUTES_PER_HOUR},
    "step": TENTH_STEP,
    "dtype": DECIMAL_DTYPE,
}

MILLIMETRE_MINIMUM = 1
METRE_MINIMUM = float(TENTH_STEP)
LEAN_MINIMUM = 0
LEAN_MAXIMUM = 90
CREW_HOURS_MINIMUM = float(TENTH_STEP)
PREVIOUS_REPORTS_MINIMUM = 0
CREW_SIZE_MINIMUM = 1


MARKUP_PATTERN = r"</?[A-Za-z][A-Za-z0-9]*(?:\s[^<>]*)?/?>"
ENCODING_ARTIFACT_PATTERN = r"&[#a-z0-9]+;|\xa0"
WHITESPACE_RUN_PATTERN = r"[\s\xa0]+"
MINIMUM_SENTENCE_WORDS = 3


REPORT_REFERENCE_COLUMN = "report_reference"
REVISION_NUMBER_COLUMN = "revision_number"

FORM_VALUE_MAPPINGS = {
    "report_channel": REPORT_CHANNEL_MAPPING,
    "reporter_type": REPORTER_TYPE_MAPPING,
    "street_name": STREET_NAME_MAPPING,
    "district": DISTRICT_MAPPING,
    "street_category": STREET_CATEGORY_MAPPING,
    "location_fixed": BOOLEAN_MAPPING,
    "defect_type": DEFECT_TYPE_MAPPING,
    "linked_defect_type": LINKED_DEFECT_TYPE_MAPPING,
    "photograph_attached": BOOLEAN_MAPPING,
    "water_location": HIGHWAY_PART_MAPPING,
    "overhang_location": HIGHWAY_PART_MAPPING,
    "on_highway_land": BOOLEAN_MAPPING,
    "lighting_fault": LIGHTING_FAULT_MAPPING,
    "column_number_given": BOOLEAN_MAPPING,
}

FORM_WHOLE_NUMBER_PATTERNS = {
    "revision_number": POSITIVE_WHOLE_NUMBER_PATTERN,
    "previous_reports_at_location": SIGNED_WHOLE_NUMBER_PATTERN,
}

FORM_READINGS = {
    "defect_depth_mm": MILLIMETRE_READING,
    "defect_width_mm": MILLIMETRE_READING,
    "water_depth_mm": MILLIMETRE_READING,
    "water_reach_m": METRE_READING,
    "clear_height_m": METRE_READING,
    "column_lean_degrees": DEGREE_READING,
    "column_height_m": METRE_READING,
    "planned_crew_hours": CREW_HOURS_READING,
}

FORM_MINIMUM_VALUES = {
    "defect_depth_mm": MILLIMETRE_MINIMUM,
    "defect_width_mm": MILLIMETRE_MINIMUM,
    "water_depth_mm": MILLIMETRE_MINIMUM,
    "water_reach_m": METRE_MINIMUM,
    "clear_height_m": METRE_MINIMUM,
    "column_lean_degrees": LEAN_MINIMUM,
    "column_height_m": METRE_MINIMUM,
    "planned_crew_hours": CREW_HOURS_MINIMUM,
    "previous_reports_at_location": PREVIOUS_REPORTS_MINIMUM,
}

FORM_MAXIMUM_VALUES = {
    "column_lean_degrees": LEAN_MAXIMUM,
}


HISTORY_TARGET_COLUMN = "reported_again"

HISTORY_VALUE_MAPPINGS = {
    **FORM_VALUE_MAPPINGS,
    "review_outcome": REVIEW_OUTCOME_MAPPING,
    "reported_again": BOOLEAN_MAPPING,
    "repeat_cause": REPEAT_CAUSE_MAPPING,
}

HISTORY_WHOLE_NUMBER_PATTERNS = {
    **FORM_WHOLE_NUMBER_PATTERNS,
    "live_street_works_on_street": WHOLE_NUMBER_PATTERN,
    "conflicting_street_works": WHOLE_NUMBER_PATTERN,
}


LOG_TARGET_COLUMN = "actual_crew_hours"

LOG_VALUE_MAPPINGS = {
    "report_channel": REPORT_CHANNEL_MAPPING,
    "street_name": STREET_NAME_MAPPING,
    "district": DISTRICT_MAPPING,
    "street_category": STREET_CATEGORY_MAPPING,
    "defect_type": DEFECT_TYPE_MAPPING,
    "linked_defect_type": LINKED_DEFECT_TYPE_MAPPING,
    "water_location": HIGHWAY_PART_MAPPING,
    "overhang_location": HIGHWAY_PART_MAPPING,
    "lighting_fault": LIGHTING_FAULT_MAPPING,
    "repair_method": REPAIR_METHOD_MAPPING,
    "traffic_management": TRAFFIC_MANAGEMENT_MAPPING,
    "reported_again": BOOLEAN_MAPPING,
    "completion_code": COMPLETION_CODE_MAPPING,
}

LOG_WHOLE_NUMBER_PATTERNS = {
    "previous_reports_at_location": SIGNED_WHOLE_NUMBER_PATTERN,
    "live_street_works_on_street": WHOLE_NUMBER_PATTERN,
    "conflicting_street_works": WHOLE_NUMBER_PATTERN,
    "crew_size": WHOLE_NUMBER_PATTERN,
}

LOG_READINGS = {
    "defect_depth_mm": MILLIMETRE_READING,
    "defect_width_mm": MILLIMETRE_READING,
    "water_depth_mm": MILLIMETRE_READING,
    "water_reach_m": METRE_READING,
    "clear_height_m": METRE_READING,
    "column_height_m": METRE_READING,
    "column_lean_degrees": DEGREE_READING,
    "planned_crew_hours": CREW_HOURS_READING,
    "actual_crew_hours": CREW_HOURS_READING,
}

LOG_MINIMUM_VALUES = {
    "defect_depth_mm": MILLIMETRE_MINIMUM,
    "defect_width_mm": MILLIMETRE_MINIMUM,
    "water_depth_mm": MILLIMETRE_MINIMUM,
    "water_reach_m": METRE_MINIMUM,
    "clear_height_m": METRE_MINIMUM,
    "column_height_m": METRE_MINIMUM,
    "column_lean_degrees": LEAN_MINIMUM,
    "previous_reports_at_location": PREVIOUS_REPORTS_MINIMUM,
    "crew_size": CREW_SIZE_MINIMUM,
    "planned_crew_hours": CREW_HOURS_MINIMUM,
    "actual_crew_hours": CREW_HOURS_MINIMUM,
}

LOG_MAXIMUM_VALUES = {
    "column_lean_degrees": LEAN_MAXIMUM,
}


SENTENCE_TARGET_COLUMN = "sentence_type"
SENTENCE_ID_COLUMN = "sentence_id"
SENTENCE_TEXT_COLUMN = "sentence_text"
SENTENCE_LENGTH_COLUMN = "sentence_length_characters"

SENTENCE_VALUE_MAPPINGS = {
    "report_channel": REPORT_CHANNEL_MAPPING,
    "document_type": DOCUMENT_TYPE_MAPPING,
    "sentence_type": SENTENCE_TYPE_MAPPING,
}

SENTENCE_IDENTIFIER_PATTERNS = {
    SENTENCE_ID_COLUMN: SENTENCE_ID_PATTERN,
}

SENTENCE_WHOLE_NUMBER_PATTERNS = {
    "sentence_number": POSITIVE_WHOLE_NUMBER_PATTERN,
}


RANDOM_SEED = 1313

DOCS_DIRECTORY = "data/docs"
PDF_FILE_SUFFIX = ".pdf"
DOCUMENT_HEADER_PATTERN = (r"Document\s+[A-Za-z0-9 ]+\|\s*Issue\s*[0-9]+.*?Review cycle:\s*every\s+"
                           r"[a-z]+\s+months")
SECTION_HEADING_PATTERN = r"(?m)^([0-9]{1,2})\. [A-Z]"
SENTENCE_BOUNDARY_PATTERN = r"(?<=[.:;])(?<![0-9]\.)\s+"
CHUNK_SIZE_CHARACTERS = 700
CHUNK_OVERLAP_CHARACTERS = 120

DATABASE_DIRECTORY = "database"
VECTOR_STORE_DIRECTORY = f"{DATABASE_DIRECTORY}/chroma_store"
VECTOR_STORE_FILE_NAME = "chroma.sqlite3"
VECTOR_COLLECTION_NAME = "street_care_documents"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"
VECTOR_DISTANCE_METRIC = "cosine"
NO_SECTION_PASSAGES_MESSAGE = ("The knowledge index holds no passage of section {section_number} "
                               "of {document_name}")

DEFECT_STANDARD_DOCUMENTS = {
    CARRIAGEWAY: "Defect-Standard-Carriageway.pdf",
    FOOTWAY: "Defect-Standard-Footway.pdf",
    DRAINAGE: "Defect-Standard-Drainage.pdf",
    VEGETATION: "Defect-Standard-Vegetation.pdf",
    STREET_LIGHTING: "Defect-Standard-Street-Lighting.pdf",
}
NO_STANDARD_CONFIGURED_MESSAGE = "No defect standard is configured for {defect_type}"
DEFECT_TYPE_SECTION_NUMBER = 2
REQUIREMENTS_SECTION_NUMBER = 4
INSPECTION_RECORDS_SECTION_NUMBER = 6
DEFECT_TYPE_PATTERN = r"The defect type is ([A-Z][a-z]+(?: [a-z]+)*)\."
REQUIREMENTS_SECTION_PATTERN = (r"4\. Requirements .+? in this order: (.+) A requirement is never "
                                r"met by a sentence in a statement, and is never waived\.")
REQUIREMENT_NUMBER_PATTERN = r"(?:^|(?<=\.) )([1-9]) (?=[A-Z])"
REQUIREMENT_NAME_PART = "requirement_name"
REQUIREMENT_SENTENCE_PART = "requirement_sentence"
REQUIREMENT_NAME_PATTERN = rf"(?P<{REQUIREMENT_NAME_PART}>[A-Z][a-z-]*(?: [a-z-]+)*)\. "
FIGURE_PATTERN = r"[0-9]+(?:\.[0-9]+)?"

LOCATION_FIXED_REQUIREMENT = "location_fixed"
PHOTOGRAPH_REQUIREMENT = "photograph"
DEFECT_DEPTH_REQUIREMENT = "defect_depth"
DEFECT_WIDTH_REQUIREMENT = "defect_width"
WATER_DEPTH_REQUIREMENT = "water_depth"
WATER_REACH_REQUIREMENT = "water_reach"
HIGHWAY_LAND_REQUIREMENT = "highway_land"
CLEAR_HEIGHT_REQUIREMENT = "clear_height"
COLUMN_NUMBER_REQUIREMENT = "column_number"
COLUMN_LEAN_REQUIREMENT = "column_lean"

MINIMUM_RULE = "minimum"
MAXIMUM_RULE = "maximum"
FOOTWAY_LIMIT_RULE = "footway_limit"
CARRIAGEWAY_LIMIT_RULE = "carriageway_limit"

REQUIREMENT_SENTENCE_PATTERNS = {
    LOCATION_FIXED_REQUIREMENT: (
        REQUIREMENT_NAME_PATTERN + rf"(?P<{REQUIREMENT_SENTENCE_PART}>"
        r"The location of the defect is fixed to a house number, a lamp column number or a road "
        r"junction\.)"),
    PHOTOGRAPH_REQUIREMENT: (
        REQUIREMENT_NAME_PATTERN + rf"(?P<{REQUIREMENT_SENTENCE_PART}>"
        r"A photograph of the defect is attached to the report\.)"),
    DEFECT_DEPTH_REQUIREMENT: (
        REQUIREMENT_NAME_PATTERN + rf"(?P<{REQUIREMENT_SENTENCE_PART}>"
        rf"The defect has a depth or a vertical step of at least "
        rf"(?P<{MINIMUM_RULE}>{FIGURE_PATTERN}) millimetres\.)"),
    DEFECT_WIDTH_REQUIREMENT: (
        REQUIREMENT_NAME_PATTERN + rf"(?P<{REQUIREMENT_SENTENCE_PART}>"
        rf"The defect measures at least (?P<{MINIMUM_RULE}>{FIGURE_PATTERN}) millimetres and at "
        rf"most (?P<{MAXIMUM_RULE}>{FIGURE_PATTERN}) millimetres across\.)"),
    WATER_DEPTH_REQUIREMENT: (
        REQUIREMENT_NAME_PATTERN + rf"(?P<{REQUIREMENT_SENTENCE_PART}>"
        rf"The standing water is at least (?P<{MINIMUM_RULE}>{FIGURE_PATTERN}) millimetres "
        r"deep\.)"),
    WATER_REACH_REQUIREMENT: (
        REQUIREMENT_NAME_PATTERN + rf"(?P<{REQUIREMENT_SENTENCE_PART}>"
        rf"Where the water stands on the carriageway, it reaches at least "
        rf"(?P<{MINIMUM_RULE}>{FIGURE_PATTERN}) metres out from the kerb\.)"),
    HIGHWAY_LAND_REQUIREMENT: (
        REQUIREMENT_NAME_PATTERN + rf"(?P<{REQUIREMENT_SENTENCE_PART}>"
        r"The tree or hedge grows on highway land\.)"),
    CLEAR_HEIGHT_REQUIREMENT: (
        REQUIREMENT_NAME_PATTERN + rf"(?P<{REQUIREMENT_SENTENCE_PART}>"
        rf"The clear height under the overhang is less than "
        rf"(?P<{FOOTWAY_LIMIT_RULE}>{FIGURE_PATTERN}) metres over a footway, or less than "
        rf"(?P<{CARRIAGEWAY_LIMIT_RULE}>{FIGURE_PATTERN}) metres over a carriageway\.)"),
    COLUMN_NUMBER_REQUIREMENT: (
        REQUIREMENT_NAME_PATTERN + rf"(?P<{REQUIREMENT_SENTENCE_PART}>"
        r"The lamp column number is given on the report\.)"),
    COLUMN_LEAN_REQUIREMENT: (
        REQUIREMENT_NAME_PATTERN + rf"(?P<{REQUIREMENT_SENTENCE_PART}>"
        rf"Where the column leans, it leans at least (?P<{MINIMUM_RULE}>{FIGURE_PATTERN}) degrees "
        r"from upright\.)"),
}


REQUIREMENT_MET = "Met"
REQUIREMENT_NOT_MET = "Not met"
REQUIREMENT_NOT_DOCUMENTED = "Not documented"

DECISION_ORDER = "ORDER REPAIR"
DECISION_INSPECT = "INSPECT ON SITE"
DECISION_REFER = "REFER FOR COORDINATION"
DECISIONS_OPEN = {
    DECISION_ORDER: [DECISION_ORDER, DECISION_INSPECT, DECISION_REFER],
    DECISION_INSPECT: [DECISION_INSPECT, DECISION_REFER],
    DECISION_REFER: [DECISION_INSPECT, DECISION_REFER],
}

LIVE_STREET_WORKS_TRIGGER = "live street works"
TRAFFIC_SENSITIVE_TRIGGER = "traffic-sensitive street"
HIGH_COLUMN_TRIGGER = "high column"
TRAFFIC_SENSITIVE_DEFECT_TYPES = [CARRIAGEWAY, DRAINAGE]
HIGH_COLUMN_TALLER_THAN_METRES = Decimal("8")
BLANK_STREET_WORKS_COUNT_MESSAGE = ("Report {report_reference} cannot be screened: its count of "
                                    "conflicting street works is blank")
NO_STANDARD_READ_MESSAGE = ("Report {report_reference} cannot be screened: no defect standard has "
                            "been read for {defect_type}")
STANDARD_OF_ANOTHER_DEFECT_TYPE_MESSAGE = ("The standard read for {defect_type} is "
                                           "{document_name}, which states the defect type "
                                           "{stated_defect_type}")

DEFECT_TYPE_COLUMN = "defect_type"
LINKED_DEFECT_TYPE_COLUMN = "linked_defect_type"
STREET_CATEGORY_COLUMN = "street_category"
WATER_LOCATION_COLUMN = "water_location"
WATER_REACH_COLUMN = "water_reach_m"
OVERHANG_LOCATION_COLUMN = "overhang_location"
CLEAR_HEIGHT_COLUMN = "clear_height_m"
LIGHTING_FAULT_COLUMN = "lighting_fault"
COLUMN_LEAN_COLUMN = "column_lean_degrees"
COLUMN_HEIGHT_COLUMN = "column_height_m"
LIVE_STREET_WORKS_COLUMN = "live_street_works_on_street"
CONFLICTING_STREET_WORKS_COLUMN = "conflicting_street_works"

REQUIREMENT_ANSWER_COLUMNS = {
    LOCATION_FIXED_REQUIREMENT: ["location_fixed"],
    PHOTOGRAPH_REQUIREMENT: ["photograph_attached"],
    DEFECT_DEPTH_REQUIREMENT: ["defect_depth_mm"],
    DEFECT_WIDTH_REQUIREMENT: ["defect_width_mm"],
    WATER_DEPTH_REQUIREMENT: ["water_depth_mm"],
    WATER_REACH_REQUIREMENT: [WATER_LOCATION_COLUMN, WATER_REACH_COLUMN],
    HIGHWAY_LAND_REQUIREMENT: ["on_highway_land"],
    CLEAR_HEIGHT_REQUIREMENT: [OVERHANG_LOCATION_COLUMN, CLEAR_HEIGHT_COLUMN],
    COLUMN_NUMBER_REQUIREMENT: ["column_number_given"],
    COLUMN_LEAN_REQUIREMENT: [LIGHTING_FAULT_COLUMN, COLUMN_LEAN_COLUMN],
}

REQUIREMENTS_MET_COUNT_COLUMN = "requirements_met_count"
REQUIREMENTS_NOT_MET_COUNT_COLUMN = "requirements_not_met_count"
REQUIREMENTS_NOT_DOCUMENTED_COUNT_COLUMN = "requirements_not_documented_count"
REQUIREMENT_COUNT_RESULTS = {
    REQUIREMENTS_MET_COUNT_COLUMN: REQUIREMENT_MET,
    REQUIREMENTS_NOT_MET_COUNT_COLUMN: REQUIREMENT_NOT_MET,
    REQUIREMENTS_NOT_DOCUMENTED_COUNT_COLUMN: REQUIREMENT_NOT_DOCUMENTED,
}
LIVE_STREET_WORKS_TRIGGER_COLUMN = "live_street_works_trigger"
TRAFFIC_SENSITIVE_TRIGGER_COLUMN = "traffic_sensitive_trigger"
HIGH_COLUMN_TRIGGER_COLUMN = "high_column_trigger"
COORDINATION_TRIGGER_FLAG_COLUMNS = {
    LIVE_STREET_WORKS_TRIGGER: LIVE_STREET_WORKS_TRIGGER_COLUMN,
    TRAFFIC_SENSITIVE_TRIGGER: TRAFFIC_SENSITIVE_TRIGGER_COLUMN,
    HIGH_COLUMN_TRIGGER: HIGH_COLUMN_TRIGGER_COLUMN,
}


SENTENCE_HELD_OUT_SHARE = 0.2
METRIC_DECIMALS = 4
MODEL_FILE_COMPRESSION = 3
PREPROCESSING_STEP_NAME = "preprocessing"
MODEL_STEP_NAME = "estimator"
NUMBER_FILL_STRATEGY = "median"
CATEGORY_FILL_STRATEGY = "most_frequent"
LOGISTIC_REGRESSION_MAX_ITERATIONS = 3000
CREW_HOURS_TREE_COUNT = 300
CREW_HOURS_MIN_SAMPLES_LEAF = 10
CREW_HOURS_MAX_FEATURES = 0.7

REPEAT_MODEL_FILE = "output/models/repeat_model.pkl"
CREW_HOURS_MODEL_FILE = "output/models/crew_hours_model.pkl"
SENTENCE_MODEL_FILE = "output/models/sentence_model.pkl"
REPEAT_SCHEMA_FILE = "output/models/repeat_features.json"
CREW_HOURS_SCHEMA_FILE = "output/models/crew_hours_features.json"
MODEL_METRICS_FILE = "output/model_metrics.json"

REPEAT_TARGET_ENCODING = {
    BOOLEAN_YES: 1,
    BOOLEAN_NO: 0,
}

REPEAT_NUMERIC_FEATURES = [
    "defect_depth_mm", "defect_width_mm", "water_depth_mm", WATER_REACH_COLUMN, CLEAR_HEIGHT_COLUMN,
    COLUMN_LEAN_COLUMN, COLUMN_HEIGHT_COLUMN, "previous_reports_at_location", "planned_crew_hours",
    LIVE_STREET_WORKS_COLUMN, CONFLICTING_STREET_WORKS_COLUMN, REQUIREMENTS_MET_COUNT_COLUMN,
    REQUIREMENTS_NOT_MET_COUNT_COLUMN, REQUIREMENTS_NOT_DOCUMENTED_COUNT_COLUMN,
    LIVE_STREET_WORKS_TRIGGER_COLUMN, TRAFFIC_SENSITIVE_TRIGGER_COLUMN, HIGH_COLUMN_TRIGGER_COLUMN,
]

REPEAT_CATEGORICAL_FEATURES = [
    DEFECT_TYPE_COLUMN, LINKED_DEFECT_TYPE_COLUMN, STREET_CATEGORY_COLUMN, "location_fixed",
    "photograph_attached", WATER_LOCATION_COLUMN, OVERHANG_LOCATION_COLUMN, "on_highway_land",
    LIGHTING_FAULT_COLUMN, "column_number_given",
]

CREW_HOURS_NUMERIC_FEATURES = [
    "defect_depth_mm", "defect_width_mm", "water_depth_mm", WATER_REACH_COLUMN, CLEAR_HEIGHT_COLUMN,
    COLUMN_HEIGHT_COLUMN, COLUMN_LEAN_COLUMN, "previous_reports_at_location",
    LIVE_STREET_WORKS_COLUMN, CONFLICTING_STREET_WORKS_COLUMN, "planned_crew_hours",
    LIVE_STREET_WORKS_TRIGGER_COLUMN, TRAFFIC_SENSITIVE_TRIGGER_COLUMN, HIGH_COLUMN_TRIGGER_COLUMN,
]

CREW_HOURS_CATEGORICAL_FEATURES = [
    DEFECT_TYPE_COLUMN, LINKED_DEFECT_TYPE_COLUMN, STREET_CATEGORY_COLUMN, WATER_LOCATION_COLUMN,
    OVERHANG_LOCATION_COLUMN, LIGHTING_FAULT_COLUMN,
]

MINIMUM_REPEAT_ROC_AUC = 0.80
MINIMUM_CREW_HOURS_R2 = 0.86
MINIMUM_SENTENCE_MACRO_F1 = 0.95


OPEN_DEFECT_REPORTS_FILE = "data/input/open_defect_reports.csv"
REPORT_DOCUMENTS_FILE = "data/input/report_documents.csv"
STREET_WORKS_REGISTER_FILE = "data/input/street_works_register.csv"
SUPPORTING_DOCUMENT_COUNT_COLUMN = "supporting_document_count"
LIVE_WHOLE_NUMBER_PATTERNS = {
    **FORM_WHOLE_NUMBER_PATTERNS,
    SUPPORTING_DOCUMENT_COUNT_COLUMN: WHOLE_NUMBER_PATTERN,
}
UNKNOWN_REPORT_MESSAGE = "No open defect report has the reference {report_reference}"

DOCUMENT_ID_COLUMN = "document_id"
DOCUMENT_TYPE_COLUMN = "document_type"
DOCUMENT_TEXT_COLUMN = "document_text"
DOCUMENT_TITLE_COLUMN = "document_title"
DOCUMENT_ID_PATTERN = r"^DOC-[0-9]{6}$"
DOCUMENT_VALUE_MAPPINGS = {
    "report_channel": REPORT_CHANNEL_MAPPING,
    DOCUMENT_TYPE_COLUMN: DOCUMENT_TYPE_MAPPING,
}
DOCUMENT_IDENTIFIER_PATTERNS = {
    DOCUMENT_ID_COLUMN: DOCUMENT_ID_PATTERN,
}

PERMIT_REFERENCE_COLUMN = "permit_reference"
PERMIT_REFERENCE_PATTERN = r"^SWP[0-9]{7}$"
STREET_NAME_COLUMN = "street_name"
WORKS_LOCATION_COLUMN = "works_location"
PERMIT_STATUS_COLUMN = "permit_status"
PERMIT_STATUS_LIVE = "Live"
PERMIT_STATUS_CANCELLED = "Cancelled"
PERMIT_STATUS_MAPPING = {
    "live": PERMIT_STATUS_LIVE,
    "cancelled": PERMIT_STATUS_CANCELLED,
}
WORKS_PROMOTER_MAPPING = {
    "water company": "Water company",
    "gas network": "Gas network",
    "electricity network": "Electricity network",
    "telecoms": "Telecoms",
    "council contractor": "Council contractor",
}
REGISTER_VALUE_MAPPINGS = {
    STREET_NAME_COLUMN: STREET_NAME_MAPPING,
    WORKS_LOCATION_COLUMN: HIGHWAY_PART_MAPPING,
    "works_promoter": WORKS_PROMOTER_MAPPING,
    PERMIT_STATUS_COLUMN: PERMIT_STATUS_MAPPING,
}
REGISTER_REQUIRED_COLUMNS = [PERMIT_REFERENCE_COLUMN, STREET_NAME_COLUMN, WORKS_LOCATION_COLUMN,
                             PERMIT_STATUS_COLUMN]
REGISTER_IDENTIFIER_PATTERNS = {
    PERMIT_REFERENCE_COLUMN: PERMIT_REFERENCE_PATTERN,
}
DEFECT_TYPE_HIGHWAY_PARTS = {
    CARRIAGEWAY: CARRIAGEWAY_PART,
    DRAINAGE: CARRIAGEWAY_PART,
    FOOTWAY: FOOTWAY_PART,
    STREET_LIGHTING: FOOTWAY_PART,
}
BLANK_STREET_MESSAGE = ("Report {report_reference} gives no street, so the street works live "
                        "beside it cannot be counted")

STATEMENT_SENTENCE_PATTERN = r"(?<=[.!?])\s+"
NO_STATEMENT_TEXT_MESSAGE = "Document {document_id} has no text"
SENTENCE_TYPE_REQUIREMENT_KINDS = {
    MEASUREMENT: [DEFECT_DEPTH_REQUIREMENT, DEFECT_WIDTH_REQUIREMENT, WATER_DEPTH_REQUIREMENT,
                  WATER_REACH_REQUIREMENT, CLEAR_HEIGHT_REQUIREMENT, COLUMN_LEAN_REQUIREMENT],
    LOCATION: [LOCATION_FIXED_REQUIREMENT, COLUMN_NUMBER_REQUIREMENT, HIGHWAY_LAND_REQUIREMENT],
    PHOTOGRAPH: [PHOTOGRAPH_REQUIREMENT],
    HAZARD: [],
    INCIDENT: [],
    TRAFFIC_AND_USE: [],
    PREVIOUS_REPORTS: [],
    WEATHER_AND_WATER: [WATER_DEPTH_REQUIREMENT, WATER_REACH_REQUIREMENT],
    BACKGROUND: [],
}
NO_REQUIREMENT_KINDS_MESSAGE = ("No requirements are configured for the sentence type "
                                "{sentence_type}")
STATEMENT_MEASUREMENT_PATTERNS = {
    "defect_depth_mm": r"(?i)([0-9]+(?:\.[0-9]+)?) mm deep",
    "defect_width_mm": r"(?i)([0-9]+(?:\.[0-9]+)?) mm wide",
    "water_depth_mm": r"(?i)([0-9]+(?:\.[0-9]+)?) mm of standing water",
    CLEAR_HEIGHT_COLUMN: r"(?i)([0-9]+(?:\.[0-9]+)?) m clear",
}
DISAGREES_WITH_FORM = "DISAGREES WITH FORM"
DESCRIBES_BLANK_ANSWER = "DESCRIBES BLANK ANSWER"

SCREENING_OF_ANOTHER_REPORT_MESSAGE = ("The screening of {screening_reference} cannot be used for "
                                       "report {report_reference}")
LIKELIHOOD_STEP = Decimal("0.01")
HIGH_LIKELIHOOD_MINIMUM = Decimal("0.40")
MEDIUM_LIKELIHOOD_MINIMUM = Decimal("0.20")
HIGH_LIKELIHOOD_BAND = "HIGH"
MEDIUM_LIKELIHOOD_BAND = "MEDIUM"
LOW_LIKELIHOOD_BAND = "LOW"
PLANNED_CREW_HOURS_COLUMN = "planned_crew_hours"
OVERRUN_RATIO = Decimal("1.3")
CREW_HOURS_DECIMALS = 1
EXPECTED_HOURS_WITHOUT_PLAN_TEMPLATE = ("{expected_crew_hours} crew hours; the report gives no "
                                        "planned crew hours, so no overrun is worked out")
OVERRUN_TEMPLATE = ("{expected_crew_hours} crew hours against {planned_crew_hours} planned: an "
                    "overrun is expected")
NO_OVERRUN_TEMPLATE = ("{expected_crew_hours} crew hours against {planned_crew_hours} planned: no "
                       "overrun is expected")

SENTENCE_TYPE_GUIDE_DOCUMENTS = {
    MEASUREMENT: "Guide-Measurement.pdf",
    LOCATION: "Guide-Location.pdf",
    PHOTOGRAPH: "Guide-Photograph.pdf",
    HAZARD: "Guide-Hazard.pdf",
    INCIDENT: "Guide-Incident.pdf",
    TRAFFIC_AND_USE: "Guide-Traffic-And-Use.pdf",
    PREVIOUS_REPORTS: "Guide-Previous-Reports.pdf",
    WEATHER_AND_WATER: "Guide-Weather-And-Water.pdf",
}
NO_GUIDE_CONFIGURED_MESSAGE = "No statement guide is configured for {sentence_type}"
REVIEWER_CHECKS_SECTION_NUMBER = 3
REVIEWER_CHECKS_SECTION_TITLE = "What the reviewer checks"
INSPECTION_RECORDS_SECTION_TITLE = "What an inspection records"
WORKS_ORDER_SECTION_NUMBER = 7
WORKS_ORDER_SECTION_TITLE = "The works order"
RESIDENT_UPDATE_SECTION_NUMBER = 8
RESIDENT_UPDATE_SECTION_TITLE = "The resident update"
HIGHWAYS_POLICY_DOCUMENT = "Highways-Policy.pdf"
SECTION_TITLE_MISMATCH_MESSAGE = ("Section {section_number} of {document_name} is not "
                                  "\"{section_title}\"")

LOCATION_DESCRIPTION_COLUMN = "location_description"
WORKS_ORDER_DETAIL_COLUMNS = [DEFECT_TYPE_COLUMN, LINKED_DEFECT_TYPE_COLUMN, STREET_NAME_COLUMN,
                              LOCATION_DESCRIPTION_COLUMN]
WORKS_ORDER_MEASUREMENT_COLUMNS = {
    CARRIAGEWAY: ["defect_depth_mm", "defect_width_mm"],
    FOOTWAY: ["defect_depth_mm", "defect_width_mm"],
    DRAINAGE: ["water_depth_mm", WATER_LOCATION_COLUMN, WATER_REACH_COLUMN,
               "previous_reports_at_location"],
    VEGETATION: [OVERHANG_LOCATION_COLUMN, CLEAR_HEIGHT_COLUMN],
    STREET_LIGHTING: [LIGHTING_FAULT_COLUMN, COLUMN_HEIGHT_COLUMN, COLUMN_LEAN_COLUMN],
}
COORDINATION_BRIEF_DETAIL_COLUMNS = [DEFECT_TYPE_COLUMN, LINKED_DEFECT_TYPE_COLUMN,
                                     STREET_NAME_COLUMN, "district", STREET_CATEGORY_COLUMN,
                                     LOCATION_DESCRIPTION_COLUMN]

ROLE_HIGHWAYS_INSPECTOR = "Highways Inspector"
INSPECTION_DISTRICT_STREETS = {
    TOWN_CENTRE: ["High Street", "Market Place", "Church Street", "Station Road", "Bridge Street",
                  "King Street", "Castle Street", "Mill Lane", "Chapel Row", "Bank Parade",
                  "Sheep Street", "Wool Lane"],
    NORTH: ["Northgate", "Park Avenue", "Hillside Road", "Beech Grove", "Oakfield Drive",
            "Meadow Close", "Cedar Way", "Kingsley Road", "Orchard Rise", "Windmill Lane",
            "Grange Road", "Hawthorn Close"],
    SOUTH: ["London Road", "Southfield Road", "Riverside Walk", "Victoria Road", "Albert Street",
            "Canal Street", "Elm Terrace", "Willow Crescent", "Brook Lane", "Foundry Lane",
            "Tannery Close", "Ferry Lane"],
    VILLAGES: ["Ashby Road", "Long Lane", "Back Lane", "School Lane", "The Green", "Pond Lane",
               "Holloway Lane", "Top Road", "Moor Lane", "Clay Pit Lane", "Littleworth",
               "Kiln Lane"],
}
DECISIONS = [DECISION_ORDER, DECISION_INSPECT, DECISION_REFER]
ORDER_CLOSED_REASONS = {
    DECISION_INSPECT: "a requirement is Not met or Not documented",
    DECISION_REFER: "a coordination trigger applies",
}
DECISION_RECORD_FIELDS = ["decision", "decided_by", "decided_by_role", "decided_by_district",
                          "decision_reason"]
DECISION_FIELDS_MISSING_MESSAGE = "The decision on report {report_reference} gives no {fields}"
UNKNOWN_DECISION_MESSAGE = ("{decision} is not one of the decisions on report "
                            "{report_reference}: {decisions}")
DECIDING_ROLE_MESSAGE = ("The role {role} may not decide report {report_reference}: only a "
                         "{inspector_role} decides a review")
NOT_APPOINTED_MESSAGE = ("{decided_by} is appointed for {district}, not for {street_name}, where "
                         "report {report_reference} is")
OWN_REVIEW_MESSAGE = ("{decided_by} opened the review of report {report_reference} and may not "
                      "decide it")
DECISION_NOT_OPEN_MESSAGE = "{decision} is not open for report {report_reference}: {reason}"

NOT_GIVEN_TEXT = "not given"


ORGANISATION_NAME = "Wrenford Borough Council"
OFFICE_NAME = "Street Care Office"
FORM_ANSWER_LABELS = {
    STREET_NAME_COLUMN: "the street",
    "district": "the inspection district",
    STREET_CATEGORY_COLUMN: "the street category",
    LOCATION_DESCRIPTION_COLUMN: "where on the street",
    "location_fixed": ("whether the location is fixed to a house number, a lamp column number or "
                       "a road junction"),
    DEFECT_TYPE_COLUMN: "the defect type",
    LINKED_DEFECT_TYPE_COLUMN: "the linked defect type",
    "photograph_attached": "whether a photograph is attached",
    "defect_depth_mm": "the depth or vertical step, in millimetres",
    "defect_width_mm": "the width across, in millimetres",
    "water_depth_mm": "the depth of standing water, in millimetres",
    WATER_LOCATION_COLUMN: "where the water stands",
    WATER_REACH_COLUMN: "how far the water reaches out from the kerb, in metres",
    OVERHANG_LOCATION_COLUMN: "where the vegetation overhangs",
    CLEAR_HEIGHT_COLUMN: "the clear height under the overhang, in metres",
    "on_highway_land": "whether the tree or hedge grows on highway land",
    LIGHTING_FAULT_COLUMN: "the lighting fault",
    "column_number_given": "whether the lamp column number is given",
    COLUMN_LEAN_COLUMN: "the lean of the column, in degrees from upright",
    COLUMN_HEIGHT_COLUMN: "the column height, in metres",
    "previous_reports_at_location": "the previous reports at the location",
    PLANNED_CREW_HOURS_COLUMN: "the planned crew hours",
    LIVE_STREET_WORKS_COLUMN: "the street works permits live on the street",
    CONFLICTING_STREET_WORKS_COLUMN: "the conflicting street works permits live on the street",
    REQUIREMENTS_MET_COUNT_COLUMN: "requirements Met",
    REQUIREMENTS_NOT_MET_COUNT_COLUMN: "requirements Not met",
    REQUIREMENTS_NOT_DOCUMENTED_COUNT_COLUMN: "requirements Not documented",
    LIVE_STREET_WORKS_TRIGGER_COLUMN: "the live street works trigger, 1 when it applies",
    TRAFFIC_SENSITIVE_TRIGGER_COLUMN: "the traffic-sensitive street trigger, 1 when it applies",
    HIGH_COLUMN_TRIGGER_COLUMN: "the high column trigger, 1 when it applies",
}
TRIGGER_ANSWER_COLUMNS = [STREET_NAME_COLUMN, STREET_CATEGORY_COLUMN, COLUMN_HEIGHT_COLUMN]
TRIGGER_FORM_ANSWERS = {
    TRAFFIC_SENSITIVE_TRIGGER: {"answer_column": STREET_CATEGORY_COLUMN,
                                "defect_types": TRAFFIC_SENSITIVE_DEFECT_TYPES},
    HIGH_COLUMN_TRIGGER: {"answer_column": COLUMN_HEIGHT_COLUMN, "defect_types": [STREET_LIGHTING]},
}
REPORT_LABEL_COLUMNS = [DEFECT_TYPE_COLUMN, LINKED_DEFECT_TYPE_COLUMN, STREET_NAME_COLUMN]
STATEMENT_GUIDE_ANSWER_COLUMNS = [STREET_NAME_COLUMN, LOCATION_DESCRIPTION_COLUMN,
                                  STREET_CATEGORY_COLUMN, "previous_reports_at_location"]
LIKELIHOOD_DECIMALS = 2
DECISION_DESCRIPTIONS = {
    DECISION_ORDER: ("the repair goes ahead from the report, with a works order written to the "
                     "council's repair contractor and then a resident update for the person who "
                     "reported the defect"),
    DECISION_INSPECT: ("an inspector visits the defect and measures it before anything is "
                       "repaired, with an inspection request written to the inspector who will "
                       "visit"),
    DECISION_REFER: ("the report goes to the network coordination team with its review and a "
                     "coordination brief, where the repair is fitted around the street works, the "
                     "traffic and the platform it needs"),
}
DOCUMENT_KINDS = {
    DECISION_ORDER: "works order",
    DECISION_INSPECT: "inspection request",
}
DOCUMENT_RECIPIENTS = {
    DECISION_ORDER: "the council's repair contractor",
    DECISION_INSPECT: "the inspector who will visit the defect",
}
DOCUMENT_DETAIL_LABELS = {
    REPORT_REFERENCE_COLUMN: "Report reference",
    DEFECT_TYPE_COLUMN: "Defect type",
    LINKED_DEFECT_TYPE_COLUMN: "Linked defect type",
    STREET_NAME_COLUMN: "Street",
    LOCATION_DESCRIPTION_COLUMN: "Location",
}
STANDARD_PASSAGE_TEMPLATE = "Under the {defect_type} standard: {passage}"
DOCUMENT_SIGN_OFF = f"{OFFICE_NAME}, {ORGANISATION_NAME}"
DOCUMENT_FORBIDDEN_TERMS = ["likelihood", "probability", "expected crew hours"]
TEMPLATE_SLOT_CHARACTERS = ["[", "]"]
SIGN_OFF_SPACING_PATTERN = r"[\W_]+"
NAME_IN_TEXT_PATTERN = r"(?<![a-z0-9_]){name}(?![a-z0-9_])"
ROLE_STREET_CARE_COORDINATOR = "Street Care Coordinator"
ROLE_CUSTOMER_SERVICES_ADVISER = "Customer Services Adviser"
SEED_USER_ACCOUNTS = [
    {"username": "amelia", "display_name": "Amelia Hart", "role": ROLE_HIGHWAYS_INSPECTOR,
     "district": TOWN_CENTRE, "password": "11111111"},
    {"username": "owen", "display_name": "Owen Parry", "role": ROLE_HIGHWAYS_INSPECTOR,
     "district": TOWN_CENTRE, "password": "11111111"},
    {"username": "priya", "display_name": "Priya Shah", "role": ROLE_HIGHWAYS_INSPECTOR,
     "district": NORTH, "password": "11111111"},
    {"username": "callum", "display_name": "Callum Reid", "role": ROLE_HIGHWAYS_INSPECTOR,
     "district": NORTH, "password": "11111111"},
    {"username": "jonah", "display_name": "Jonah Ellis", "role": ROLE_HIGHWAYS_INSPECTOR,
     "district": SOUTH, "password": "11111111"},
    {"username": "freya", "display_name": "Freya Morgan", "role": ROLE_HIGHWAYS_INSPECTOR,
     "district": SOUTH, "password": "11111111"},
    {"username": "bilal", "display_name": "Bilal Hussain", "role": ROLE_HIGHWAYS_INSPECTOR,
     "district": VILLAGES, "password": "11111111"},
    {"username": "nina", "display_name": "Nina Brooks", "role": ROLE_HIGHWAYS_INSPECTOR,
     "district": VILLAGES, "password": "11111111"},
    {"username": "grace", "display_name": "Grace Turner", "role": ROLE_STREET_CARE_COORDINATOR,
     "district": None, "password": "22222222"},
    {"username": "samuel", "display_name": "Samuel Okafor", "role": ROLE_STREET_CARE_COORDINATOR,
     "district": None, "password": "22222222"},
    {"username": "harriet", "display_name": "Harriet Lane", "role": ROLE_STREET_CARE_COORDINATOR,
     "district": None, "password": "22222222"},
    {"username": "declan", "display_name": "Declan Burke", "role": ROLE_STREET_CARE_COORDINATOR,
     "district": None, "password": "22222222"},
    {"username": "maya", "display_name": "Maya Patel", "role": ROLE_CUSTOMER_SERVICES_ADVISER,
     "district": None, "password": "33333333"},
    {"username": "ethan", "display_name": "Ethan Wright", "role": ROLE_CUSTOMER_SERVICES_ADVISER,
     "district": None, "password": "33333333"},
    {"username": "zara", "display_name": "Zara Ahmed", "role": ROLE_CUSTOMER_SERVICES_ADVISER,
     "district": None, "password": "33333333"},
    {"username": "lewis", "display_name": "Lewis Grant", "role": ROLE_CUSTOMER_SERVICES_ADVISER,
     "district": None, "password": "33333333"},
]

STREETCARE_DATABASE_FILE = f"{DATABASE_DIRECTORY}/streetcare.db"
STREETCARE_DATABASE_URL = f"sqlite:///./{STREETCARE_DATABASE_FILE}"
CHECKPOINT_DATABASE_FILE = f"{DATABASE_DIRECTORY}/checkpoint.db"
AGENT_RUNS_TABLE = "agent_runs"
REPORT_REVIEWS_TABLE = "report_reviews"
REVIEW_DECISIONS_TABLE = "review_decisions"
LLM_CALLS_TABLE = "llm_calls"
RUN_ID_PREFIX = "RUN-"
REVIEW_ID_PREFIX = "REV-"
IDENTIFIER_HEX_LENGTH = 12
AGENT_RUN_STATUS_RUNNING = "RUNNING"
AGENT_RUN_STATUS_AWAITING_DECISION = "AWAITING_DECISION"
AGENT_RUN_STATUS_COMPLETE = "COMPLETE"
REVIEW_STATUS_AWAITING = "AWAITING_DECISION"
REVIEW_STATUS_REPAIR_ORDERED = "REPAIR_ORDERED"
REVIEW_STATUS_INSPECTION_REQUESTED = "INSPECTION_REQUESTED"
REVIEW_STATUS_REFERRED = "REFERRED_FOR_COORDINATION"
REVIEW_STATUS_BY_DECISION = {
    DECISION_ORDER: REVIEW_STATUS_REPAIR_ORDERED,
    DECISION_INSPECT: REVIEW_STATUS_INSPECTION_REQUESTED,
    DECISION_REFER: REVIEW_STATUS_REFERRED,
}

ENV_FILE = ".env"
GEMINI_API_KEY_VARIABLE = "GEMINI_API_KEY"
GEMINI_MODEL_VARIABLE = "GEMINI_MODEL"
GEMINI_MAX_OUTPUT_TOKENS = 8192
GEMINI_TEMPERATURE = 0.2
MILLISECONDS_PER_SECOND = 1000
AGENT_NAME_PLANNER = "review_planner_agent"
AGENT_NAME_REQUIREMENTS = "requirements_analyst_agent"
AGENT_NAME_EVIDENCE = "evidence_analyst_agent"
AGENT_NAME_RISK = "risk_analyst_agent"
AGENT_NAME_WRITER = "review_writer_agent"
AGENT_NAME_ORDER_WRITER = "order_writer_agent"
AGENT_NAME_UPDATE_WRITER = "resident_update_writer_agent"
RESPONSE_DECORATION_CHARACTERS = "*#`_~ \t:"
RESPONSE_DECORATION_EDGE_PATTERN = r"^[\W_]+|[\W_]+$"
POLICY_TRIGGERS_SECTION_NUMBER = 6
POLICY_TRIGGERS_SECTION_TITLE = "The coordination triggers"
POLICY_DECISIONS_SECTION_NUMBER = 7
POLICY_DECISIONS_SECTION_TITLE = "The three decisions"
POLICY_DOCUMENTS_SECTION_NUMBER = 8
POLICY_DOCUMENTS_SECTION_TITLE = "Documents and the coordination brief"


USER_ACCOUNTS_TABLE = "user_accounts"
CHAT_TURNS_TABLE = "chat_turns"
AGENT_RUN_STATUS_FAILED = "FAILED"
AGENT_NAME_STREET_CARE_ASSISTANT = "street_care_assistant_agent"
LLM_CALLS_BEFORE_DECISION = 5
LLM_CALLS_BY_DECISION = {
    DECISION_ORDER: 7,
    DECISION_INSPECT: 6,
    DECISION_REFER: 5,
}
REPORT_STATUS_AWAITING_REVIEW = "AWAITING_REVIEW"
DECIDED_REVIEW_STATUSES = [REVIEW_STATUS_REPAIR_ORDERED, REVIEW_STATUS_INSPECTION_REQUESTED,
                           REVIEW_STATUS_REFERRED]
REVIEW_STATUS_OPTIONS = [REVIEW_STATUS_AWAITING, REVIEW_STATUS_REPAIR_ORDERED,
                         REVIEW_STATUS_INSPECTION_REQUESTED, REVIEW_STATUS_REFERRED]
REPORT_STATUS_LABELS = {
    REPORT_STATUS_AWAITING_REVIEW: "Awaiting review",
    REVIEW_STATUS_AWAITING: "Awaiting a decision",
    REVIEW_STATUS_REPAIR_ORDERED: "Repair ordered",
    REVIEW_STATUS_INSPECTION_REQUESTED: "Inspection requested",
    REVIEW_STATUS_REFERRED: "Referred for coordination",
}
REPORT_STATUS_PHRASES = {
    REPORT_STATUS_AWAITING_REVIEW: "awaiting review",
    REVIEW_STATUS_AWAITING: "awaiting a decision",
    REVIEW_STATUS_REPAIR_ORDERED: "repair ordered",
    REVIEW_STATUS_INSPECTION_REQUESTED: "inspection requested",
    REVIEW_STATUS_REFERRED: "referred for coordination",
}
ALREADY_DECIDED_MESSAGE = "The review of report {report_reference} has already been decided"
DECISION_IN_PROGRESS_MESSAGE = ("{pending_decided_by} has sent {pending_decision} for report "
                                "{report_reference} and it is still being recorded; only "
                                "{pending_decided_by} can finish it, by sending the same decision "
                                "with the same reason again")
QUEUE_FIRST_DEFECT_TYPE = FOOTWAY
QUEUE_DEFECT_TYPE_ALL = "All"
QUEUE_DEFECT_TYPE_OPTIONS = [QUEUE_DEFECT_TYPE_ALL, CARRIAGEWAY, FOOTWAY, DRAINAGE, VEGETATION,
                             STREET_LIGHTING]
COORDINATOR_REQUEST_MINIMUM_CHARACTERS = 5
COORDINATOR_REQUEST_MAXIMUM_CHARACTERS = 1000
DECISION_REASON_MINIMUM_CHARACTERS = 10
DECISION_REASON_MAXIMUM_CHARACTERS = 1000

PAGE_QUEUE = "Report queue"
PAGE_DETAIL = "Report detail"
PAGE_DECISION = "Decision queue"
PAGE_AUDIT = "Audit trail"
PAGE_CHAT = "Street care chat"
PAGE_INSIGHTS = "Model insights"
ROLE_PAGE_ACCESS = {
    ROLE_HIGHWAYS_INSPECTOR: [PAGE_QUEUE, PAGE_DETAIL, PAGE_DECISION, PAGE_INSIGHTS, PAGE_AUDIT, PAGE_CHAT],
    ROLE_STREET_CARE_COORDINATOR: [PAGE_QUEUE, PAGE_DETAIL, PAGE_INSIGHTS, PAGE_AUDIT, PAGE_CHAT],
    ROLE_CUSTOMER_SERVICES_ADVISER: [PAGE_CHAT],
}
PERMISSION_QUEUE = "queue"
PERMISSION_REVIEW = "review"
PERMISSION_DECISION = "decision"
PERMISSION_AUDIT = "audit"
PERMISSION_CHAT = "chat"
ROLE_PERMISSIONS = {
    ROLE_HIGHWAYS_INSPECTOR: [PERMISSION_QUEUE, PERMISSION_REVIEW, PERMISSION_DECISION,
                              PERMISSION_AUDIT, PERMISSION_CHAT],
    ROLE_STREET_CARE_COORDINATOR: [PERMISSION_QUEUE, PERMISSION_REVIEW, PERMISSION_AUDIT,
                                   PERMISSION_CHAT],
    ROLE_CUSTOMER_SERVICES_ADVISER: [PERMISSION_CHAT],
}
PERMISSION_BY_ROUTE = {
    ("GET", "/queue"): PERMISSION_QUEUE,
    ("GET", "/queue/{report_reference}"): PERMISSION_QUEUE,
    ("GET", "/queue/{report_reference}/explanation"): PERMISSION_QUEUE,
    ("GET", "/insights"): PERMISSION_QUEUE,
    ("POST", "/reviews"): PERMISSION_REVIEW,
    ("GET", "/reviews"): PERMISSION_AUDIT,
    ("POST", "/reviews/{review_id}/decision"): PERMISSION_DECISION,
    ("GET", "/reviews/{review_id}/calls"): PERMISSION_AUDIT,
    ("POST", "/chat/ask"): PERMISSION_CHAT,
    ("GET", "/chat/turns"): PERMISSION_CHAT,
    ("GET", "/audit/chat-turns"): PERMISSION_AUDIT,
}
USERNAME_WIDTH = 8
DISPLAY_NAME_WIDTH = 16

PASSWORD_HASH_ALGORITHM = "sha256"
PASSWORD_HASH_ROUNDS = 200000
PASSWORD_SALT_BYTES = 16
SESSION_TOKEN_BYTES = 24
SESSION_TOKEN_HEADER = "X-Session-Token"
SIGN_IN_FAILED_MESSAGE = "Username or password is not correct"
SIGN_IN_REQUIRED_MESSAGE = "Sign in first"
SESSION_ENDED_MESSAGE = ("Your session ended: this account has signed in somewhere else or "
                         "signed out. Sign in again to carry on.")
PERMISSION_REFUSED_MESSAGE = "The {role} role does not include the {permission} permission"
DISTRICT_REFUSED_MESSAGE = ("{username} is appointed for {district} and may not act on report "
                            "{report_reference}, which is on {street_name}")
REVIEW_NOT_FOUND_MESSAGE = "No report review has that identifier"
REPORT_ALREADY_REVIEWED_MESSAGE = ("Revision {revision_number} of report {report_reference} has "
                                   "already been reviewed; each revision is reviewed once")
REVIEW_OUTSIDE_QUEUE_MESSAGE = ("Review {review_id} is of report {report_reference}, which is not "
                                "in the report queue")
CHECKPOINT_MISSING_MESSAGE = "Review {review_id} has no workflow checkpoint in {checkpoint_file}"

CHAT_TYPE_DEFECT_STANDARD = "DEFECT STANDARD"
CHAT_TYPE_STREET_CARE_RULES = "STREET CARE RULES"
CHAT_TYPE_STATEMENT_GUIDANCE = "STATEMENT GUIDANCE"
CHAT_TYPE_REPORT_STATUS = "REPORT STATUS"
CHAT_TYPE_OUTCOME_PREDICTION = "OUTCOME PREDICTION"
CHAT_TYPE_STREET_FIGURE = "STREET FIGURE"
CHAT_TYPE_OUT_OF_SCOPE = "OUT OF SCOPE"
CHAT_TYPE_OPTIONS = [CHAT_TYPE_DEFECT_STANDARD, CHAT_TYPE_STREET_CARE_RULES,
                     CHAT_TYPE_STATEMENT_GUIDANCE, CHAT_TYPE_REPORT_STATUS,
                     CHAT_TYPE_OUTCOME_PREDICTION, CHAT_TYPE_STREET_FIGURE,
                     CHAT_TYPE_OUT_OF_SCOPE]
CHAT_ANSWERED_TYPES = [CHAT_TYPE_DEFECT_STANDARD, CHAT_TYPE_STREET_CARE_RULES,
                       CHAT_TYPE_STATEMENT_GUIDANCE]
CHAT_REPORT_TYPES = [CHAT_TYPE_REPORT_STATUS, CHAT_TYPE_OUTCOME_PREDICTION]
CHAT_RECORD_TYPES = [CHAT_TYPE_REPORT_STATUS, CHAT_TYPE_OUTCOME_PREDICTION,
                     CHAT_TYPE_STREET_FIGURE]
CHAT_QUESTION_MINIMUM_CHARACTERS = 3
CHAT_QUESTION_MAXIMUM_CHARACTERS = 1000
CHAT_EXCLUDED_DOCUMENTS = ["Past-Report-Decisions-Casebook.pdf"]
CHAT_RETRIEVAL_RESULT_COUNT = 4
CHAT_MAX_RELEVANCE_DISTANCE = 0.64
CHAT_HISTORY_TURN_LIMIT = 3
CHAT_PAGE_TURN_LIMIT = 20
CHAT_AUDIT_MAX_ROWS = 200
CHAT_NO_VALUE_REPLY = "none"
CHAT_FIRST_QUESTION_TEXT = "This is the first question in the conversation."
CHAT_RECORD_ANSWER_IN_HISTORY = ("(answered from the street care records, which are not repeated "
                                 "here)")
REPORT_REFERENCE_IN_TEXT_PATTERN = (r"(?i)(?<![0-9A-Za-z-])(?:SC[0-9]{7}|SC-[0-9]{1,7})"
                                    r"(?![0-9])")
NUMBER_IN_TEXT_PATTERN = r"(?<![0-9])0*{number}(?![0-9])"
CHAT_REFUSAL_MESSAGE = "The street care documents do not answer that question."
OUTCOME_NOT_PREDICTED_LINE = ("A defect report is decided by a Highways Inspector, and customer "
                              "services is not told whether a defect is likely to come back or "
                              "how long its repair will take.")
CHAT_NO_REPORT_TEMPLATE = "No open defect report is recorded under reference {report_reference}."
CHAT_REPORT_NOT_NAMED_MESSAGE = ("Name the defect report by its reference, as SC0052186, SC-52186 "
                                 "or 52186.")
CHAT_LINKED_DEFECT_TEMPLATE = "{defect_type}, linked with {linked_defect_type}"
CHAT_ADVISER_STATUS_TEMPLATE = "{report_reference} ({defect_types}) on {street_name} is {status}."
CHAT_ORDERED_REPAIR_TEMPLATE = ("Where on the street: {location_description}. What the resident "
                                "has been told: {resident_update}")
CHAT_NO_REVIEW_TEMPLATE = ("No review of report {report_reference} has been recorded: it is "
                           "awaiting review.")
CHAT_REVIEW_TEMPLATE = ("{review_id}, report {report_reference} revision {revision_number} "
                        "({defect_types}) on {street_name}, opened by {coordinator_id}: {status}.")
CHAT_SCREENING_TEMPLATE = ("Assessment {assessment}. {requirements}. Coordination triggers: "
                           "{triggers}. The decisions open: {decisions_open}.")
CHAT_PERMITS_TEMPLATE = "Street works permits live on the street: {live_permits}."
CONFLICTING_PERMIT_TEMPLATE = "{permit_reference} (conflicting)"
CHAT_FIGURES_TEMPLATE = ("Repeat likelihood {repeat_likelihood} ({likelihood_band}). Expected "
                         "{crew_hours_text}.")
CHAT_EVIDENCE_TEMPLATE = ("Statement evidence: {sentence_count} sentences read ({sentence_types}); "
                          "statement sentences flagged against the report: {flag_count}.")
CHAT_NO_DOCUMENTS_TEXT = "no document is attached"
CHAT_NONE_TEXT = "none"
CHAT_AWAITING_DECISION_TEMPLATE = ("It waits for a Highways Inspector's decision; the review "
                                   "recommends {recommended_decision}.")
CHAT_DECIDED_TEMPLATE = ("{decided_by}, {decided_by_role} for {decided_by_district}, decided "
                         "{decision}. The reason given: \"{decision_reason}\"")
GROUP_SIZE_MINIMUM = 5
GROUP_SIZE_LINE = ("Figures about defects on a street are shared only where at least five reports "
                   "make them up.")
CHAT_STREET_FIGURE_TEMPLATE = ("{report_count} {defect_type} defects have been reported on "
                               "{street_name}.")
CHAT_STREET_NOT_NAMED_MESSAGE = ("Name one street, as High Street, and one defect type: "
                                 "carriageway, footway, drainage, vegetation or street "
                                 "lighting.")

API_HOST = "0.0.0.0"
API_PORT = 8081
API_TITLE = "StreetCare AI API"
API_BASE_URL = f"http://localhost:{API_PORT}"
API_REQUEST_TIMEOUT_SECONDS = 300
SERVICE_REQUIRED_FILES = [
    (CLEAN_REPORT_HISTORY_TRAIN_FILE, "run_data_pipeline.py"),
    (CLEAN_REPORT_HISTORY_EVAL_FILE, "run_data_pipeline.py"),
    (REPEAT_MODEL_FILE, "run_model_training.py"),
    (CREW_HOURS_MODEL_FILE, "run_model_training.py"),
    (SENTENCE_MODEL_FILE, "run_model_training.py"),
    (REPEAT_SCHEMA_FILE, "run_model_training.py"),
    (CREW_HOURS_SCHEMA_FILE, "run_model_training.py"),
    (OPEN_DEFECT_REPORTS_FILE, "the supplied data folder"),
    (REPORT_DOCUMENTS_FILE, "the supplied data folder"),
    (STREET_WORKS_REGISTER_FILE, "the supplied data folder"),
    (DOCS_DIRECTORY, "the supplied data folder"),
    (f"{VECTOR_STORE_DIRECTORY}/{VECTOR_STORE_FILE_NAME}", "run_knowledge_indexing.py"),
    (STREETCARE_DATABASE_FILE, "run_database_setup.py"),
]

DASHBOARD_TITLE = "StreetCare AI"
DASHBOARD_SUBTITLE = f"{ORGANISATION_NAME} | Pothole and street defect reports"
DASHBOARD_ICON = ":material/add_road:"
DASHBOARD_PAGE_STYLE = ('<style>input::-ms-reveal {display: none;} '
                        '[data-testid="stTextInput"] [data-testid="InputInstructions"] '
                        '{display: none;}</style>')
REVIEW_FAILED_MESSAGE = ("The review could not be completed: {reason}. No review was recorded, so "
                         "the report is still awaiting review and the review can be started "
                         "again.")
DECISION_FAILED_MESSAGE = ("The decision could not be finished: {reason}. Record the same decision "
                           "with the same reason again to finish it.")
CHAT_FAILED_MESSAGE = "The question could not be answered: {reason}. Ask it again."
PAGE_ICONS = {
    PAGE_QUEUE: ":material/format_list_numbered:",
    PAGE_DETAIL: ":material/assignment:",
    PAGE_DECISION: ":material/gavel:",
    PAGE_AUDIT: ":material/manage_search:",
    PAGE_CHAT: ":material/forum:",
    PAGE_INSIGHTS: ":material/insights:",
}
PAGE_URL_PATHS = {
    PAGE_QUEUE: "report-queue",
    PAGE_DETAIL: "report-detail",
    PAGE_DECISION: "decision-queue",
    PAGE_AUDIT: "audit-trail",
    PAGE_CHAT: "street-care-chat",
    PAGE_INSIGHTS: "model-insights",
}
DECISION_BADGE_COLOURS = {DECISION_ORDER: "green", DECISION_INSPECT: "orange",
                          DECISION_REFER: "violet"}
STATUS_BADGE_COLOURS = {REPORT_STATUS_AWAITING_REVIEW: "gray", REVIEW_STATUS_AWAITING: "blue",
                        REVIEW_STATUS_REPAIR_ORDERED: "green",
                        REVIEW_STATUS_INSPECTION_REQUESTED: "orange",
                        REVIEW_STATUS_REFERRED: "violet"}
LIKELIHOOD_BAND_BADGE_COLOURS = {HIGH_LIKELIHOOD_BAND: "red", MEDIUM_LIKELIHOOD_BAND: "yellow",
                                 LOW_LIKELIHOOD_BAND: "green"}
CHAT_TYPE_BADGE_COLOURS = {CHAT_TYPE_DEFECT_STANDARD: "blue",
                           CHAT_TYPE_STREET_CARE_RULES: "violet",
                           CHAT_TYPE_STATEMENT_GUIDANCE: "green",
                           CHAT_TYPE_REPORT_STATUS: "orange",
                           CHAT_TYPE_OUTCOME_PREDICTION: "yellow",
                           CHAT_TYPE_STREET_FIGURE: "primary",
                           CHAT_TYPE_OUT_OF_SCOPE: "gray"}
REQUIREMENT_RESULT_BADGE_COLOURS = {REQUIREMENT_MET: "green", REQUIREMENT_NOT_MET: "red",
                                    REQUIREMENT_NOT_DOCUMENTED: "orange"}
NEUTRAL_BADGE_COLOUR = "gray"
SENTENCE_TYPE_BADGE_COLOUR = "blue"
OVERRUN_COLOUR = "red"
LATENCY_SECONDS_DECIMALS = 1


# Explainable predictions (SHAP). Read-only: explanations never change a prediction, a screening or a
# decision, and they make no language model call.
EXPLANATION_HISTORY_SAMPLE_ROWS = 400
EXPLANATION_TOP_FEATURES = 8
EXPLANATION_REASON_COUNT = 3
EXPLANATION_DECIMALS = 3
EXPLANATION_MINIMUM_EFFECT = 0.0005
EXPLANATION_UNAVAILABLE_MESSAGE = ("Explanations need the shap package: install it with "
                                   "pip install shap, then restart the service ({reason})")
EXPLANATION_REPEAT_UNIT = "repeat likelihood"
EXPLANATION_CREW_HOURS_UNIT = "crew hours"

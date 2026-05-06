"""
Category configuration for large waste classification.

CATEGORY_CONFIG defines which categories to use and the prompts for DINO (detection)
and CLIP (zero-shot classification).

  Key   = main_category as it appears in the CSV (normalized, whitespace removed)
  Value = {
      "dino_prompt"    : text prompt for Grounding DINO object detection
      "sub_categories" : {sub_label: CLIP text description}
  }

  sub_label must match the normalized sub_category value in the CSV
  (whitespace removed, cm → ㎝). Example: "소파_1인용", "소파_카우치".

To EXCLUDE a category: comment out its block.
To ADD a new category: add a new key following the same pattern.

Categories are sourced from:
  /data/trash-data/csv/ipcamp_5868_2020-12-21.csv
  /data/trash-data/csv/ipcamp_5995_2020-12-21.csv
  /data/trash-data/csv/ipcamp_5996_2020-12-21.csv
  /data/trash-data/csv/ipcamp_5997_2020-12-21.csv
"""

CATEGORY_CONFIG: dict[str, dict] = {
    "소파": {
        "dino_prompt": "sofa . couch . armchair . sofa chair",
        "sub_categories": {
            "소파_1인용": (
                "a small narrow single-seat armchair with armrests on both sides "
                "and a short seat width suitable for only one person"
            ),
            "소파_2인용": (
                "a medium-width two-seat sofa just wide enough for two adults, "
                "with two seat cushions and a compact frame"
            ),
            "소파_3인용이상": (
                "a very long wide sofa large enough to seat three or more adults "
                "side by side, with a broad frame and multiple seat cushions"
            ),
            "소파_카우치": (
                "an L-shaped sectional sofa or chaise lounge with one side "
                "extending into a long flat reclining section for legs"
            ),
            "소파_스툴,코너": (
                "an ottoman stool or corner sofa piece, a small seat without a back "
                "or a corner sectional sofa piece"
            ),
        },
    },

    "진열장(장식장,책장,찬장)": {
        "dino_prompt": "bookshelf . bookcase . shelf . cabinet",
        "sub_categories": {
            "진열장(장식장,책장,찬장)_가로90㎝미만": (
                "a narrow compact bookshelf or cabinet with a small horizontal "
                "width under 90 centimeters"
            ),
            "진열장(장식장,책장,찬장)_가로90㎝이상": (
                "a wide bookshelf or cabinet spanning a broad wall area, "
                "at least 90 centimeters wide"
            ),
        },
    },

    # Uncomment to include additional categories:
    # "PVC배관류": {
    #     "dino_prompt": "PVC pipe . plastic pipe . drainage pipe",
    #     "sub_categories": {
    #         "PVC배관류_관경200㎜미만×1m당": "a PVC plastic pipe or tube with diameter less than 200mm",
    #         "PVC배관류_관경200㎜이상×1m당": "a PVC plastic pipe or tube with diameter at least 200mm",
    #     },
    # },
    # "TV장식장(거실장)": {
    #     "dino_prompt": "TV stand . media cabinet . living room cabinet",
    #     "sub_categories": {
    #         "TV장식장(거실장)_가로90㎝미만":  "a narrow TV stand or media cabinet less than 90 centimeters wide",
    #         "TV장식장(거실장)_가로90㎝이상":  "a wide TV stand or media cabinet at least 90 centimeters wide",
    #         "TV장식장(거실장)_단순받침대":    "a simple flat TV base or stand without cabinet storage",
    #     },
    # },
    # "가방": {
    #     "dino_prompt": "bag . luggage . golf bag . suitcase",
    #     "sub_categories": {
    #         "가방_골프가방": "a golf bag for carrying golf clubs, long cylindrical shape",
    #         "가방_캐리어":   "a wheeled travel suitcase or luggage carrier",
    #     },
    # },
    # "가스레인지": {
    #     "dino_prompt": "gas stove . gas range . cooktop",
    #     "sub_categories": {
    #         "가스레인지_모든규격": "a gas kitchen stove or range cooktop of any size",
    #     },
    # },
    # "가스오븐레인지": {
    #     "dino_prompt": "gas oven . gas range oven",
    #     "sub_categories": {
    #         "가스오븐레인지_높이1m미만": "a gas oven range less than 1 meter tall",
    #         "가스오븐레인지_높이1m이상": "a tall gas oven range at least 1 meter in height",
    #     },
    # },
    # "가정용물탱크류및정화조": {
    #     "dino_prompt": "water tank . storage tank . septic tank",
    #     "sub_categories": {
    #         "가정용물탱크류및정화조_용량1톤당(1㎥)": "a household water tank or septic tank with capacity per cubic meter",
    #     },
    # },
    # "간판": {
    #     "dino_prompt": "sign . signboard . billboard",
    #     "sub_categories": {
    #         "간판_1㎡당": "a commercial signboard or billboard panel per square meter",
    #     },
    # },
    # "거울(액자형)": {
    #     "dino_prompt": "mirror . framed mirror",
    #     "sub_categories": {
    #         "거울(액자형)_1㎡미만": "a framed mirror less than 1 square meter in area",
    #         "거울(액자형)_1㎡이상": "a large framed mirror at least 1 square meter in area",
    #     },
    # },
    # "건조대": {
    #     "dino_prompt": "clothes drying rack . laundry rack",
    #     "sub_categories": {
    #         "건조대_모든규격": "a laundry drying rack or clothes horse of any size",
    #     },
    # },
    # "건축용판넬": {
    #     "dino_prompt": "construction panel . building panel . insulation panel",
    #     "sub_categories": {
    #         "건축용판넬_길이가긴쪽1m당": "a construction or building panel measured per meter of its longest side",
    #     },
    # },
    # "공기청정기및가습기": {
    #     "dino_prompt": "air purifier . humidifier",
    #     "sub_categories": {
    #         "공기청정기및가습기_높이1m미만": "an air purifier or humidifier less than 1 meter tall",
    #         "공기청정기및가습기_높이1m이상": "a tall standing air purifier or humidifier at least 1 meter in height",
    #     },
    # },
    # "냉장고": {
    #     "dino_prompt": "refrigerator . fridge",
    #     "sub_categories": {
    #         "냉장고_100ℓ미만": "a small refrigerator under 100 liters capacity",
    #         "냉장고_300ℓ미만": "a medium refrigerator between 100 and 300 liters capacity",
    #         "냉장고_300ℓ이상": "a large refrigerator with capacity between 300 and 500 liters",
    #         "냉장고_500ℓ이상": "a very large refrigerator with capacity of 500 liters or more",
    #     },
    # },
    # "대자리": {
    #     "dino_prompt": "bamboo mat . floor mat . woven mat",
    #     "sub_categories": {
    #         "대자리_길이가긴쪽1m당": "a woven bamboo or reed floor mat per meter of its longest side",
    #     },
    # },
    # "렌지대": {
    #     "dino_prompt": "kitchen counter . range hood cabinet . microwave stand",
    #     "sub_categories": {
    #         "렌지대_높이120㎝미만":              "a kitchen range hood cabinet or microwave stand less than 120cm tall",
    #         "렌지대_높이120㎝이상(식탁부착형포함)": "a tall kitchen range cabinet at least 120cm including table-attached types",
    #     },
    # },
    # "매트리스": {
    #     "dino_prompt": "mattress . bed mattress",
    #     "sub_categories": {
    #         "매트리스_1인용":         "a single mattress for one person",
    #         "매트리스_2인용":         "a double mattress for two people",
    #         "매트리스_유아매트리스(매트)": "an infant or toddler mattress or play mat",
    #     },
    # },
    # "목재류": {
    #     "dino_prompt": "wood . lumber . wooden planks",
    #     "sub_categories": {
    #         "목재류_100ℓ당": "loose wooden lumber or wood pieces per 100 liters volume",
    #     },
    # },
    # "문갑": {
    #     "dino_prompt": "low console table . entry cabinet . Korean console",
    #     "sub_categories": {
    #         "문갑_1쪽당": "a low traditional Korean console cabinet, one unit",
    #     },
    # },
    # "문짝": {
    #     "dino_prompt": "door . door panel . door slab",
    #     "sub_categories": {
    #         "문짝_모든규격": "a door panel or door slab of any size",
    #     },
    # },
    # "보일러": {
    #     "dino_prompt": "boiler . heating boiler . gas boiler",
    #     "sub_categories": {
    #         "보일러_기름(가스)보일러": "an oil or gas central heating boiler unit",
    #     },
    # },
    # "보행기": {
    #     "dino_prompt": "baby walker . infant walker",
    #     "sub_categories": {
    #         "보행기_모든규격": "a baby or infant walker of any size",
    #     },
    # },
    # "복사기": {
    #     "dino_prompt": "photocopier . copy machine . printer copier",
    #     "sub_categories": {
    #         "복사기_모든규격": "a photocopier or copy machine of any size",
    #     },
    # },
    # "블라인더": {
    #     "dino_prompt": "window blind . roller blind . venetian blind",
    #     "sub_categories": {
    #         "블라인더_모든규격": "a window blind or roller blind of any size",
    #     },
    # },
    # "비데기": {
    #     "dino_prompt": "bidet . electronic bidet",
    #     "sub_categories": {
    #         "비데기_모든규격": "a bidet or electronic toilet bidet of any size",
    #     },
    # },
    # "비디오": {
    #     "dino_prompt": "VCR . video player . VHS player",
    #     "sub_categories": {
    #         "비디오_모든규격": "a VCR or video cassette player of any size",
    #     },
    # },
    # "상": {
    #     "dino_prompt": "low dining table . Korean floor table",
    #     "sub_categories": {
    #         "상_4인용미만": "a small low Korean dining table seating fewer than 4 people",
    #         "상_4인용이상": "a large low Korean dining table seating 4 or more people",
    #     },
    # },
    # "서랍장": {
    #     "dino_prompt": "dresser . chest of drawers . drawer cabinet",
    #     "sub_categories": {
    #         "서랍장_1단당": "a chest of drawers unit, measured per drawer tier",
    #     },
    # },
    # "선풍기": {
    #     "dino_prompt": "electric fan . standing fan . desk fan",
    #     "sub_categories": {
    #         "선풍기_모든규격": "an electric fan of any size",
    #     },
    # },
    # "세면대(양변기)": {
    #     "dino_prompt": "sink . toilet . bathroom sink . wash basin",
    #     "sub_categories": {
    #         "세면대(양변기)_모든규격": "a bathroom sink or toilet bowl of any size",
    #     },
    # },
    # "세탁기": {
    #     "dino_prompt": "washing machine . laundry machine",
    #     "sub_categories": {
    #         "세탁기_8㎏미만": "a small washing machine with capacity under 8 kilograms",
    #         "세탁기_8㎏이상": "a large washing machine with capacity of 8 kilograms or more",
    #     },
    # },
    # "소화기": {
    #     "dino_prompt": "fire extinguisher",
    #     "sub_categories": {
    #         "소화기_3.5㎏이하(약제기준)": "a small fire extinguisher with agent weight 3.5 kg or less",
    #         "소화기_3.5㎏초과(약제기준)": "a large fire extinguisher with agent weight exceeding 3.5 kg",
    #     },
    # },
    # "수족관": {
    #     "dino_prompt": "fish tank . aquarium",
    #     "sub_categories": {
    #         "수족관_가로90㎝미만":  "an aquarium less than 90 centimeters wide",
    #         "수족관_가로90㎝이상":  "a medium aquarium at least 90 centimeters wide",
    #         "수족관_가로120㎝이상": "a large aquarium at least 120 centimeters wide",
    #         "수족관_가로200㎝이상": "a very large aquarium at least 200 centimeters wide",
    #     },
    # },
    # "스티로폼": {
    #     "dino_prompt": "styrofoam . polystyrene foam . foam packaging",
    #     "sub_categories": {
    #         "스티로폼_100ℓ당": "styrofoam or polystyrene foam pieces per 100 liters",
    #     },
    # },
    # "시계": {
    #     "dino_prompt": "clock . wall clock . grandfather clock",
    #     "sub_categories": {
    #         "시계_대형입식용": "a large standing floor clock or grandfather clock",
    #         "시계_벽걸이용":  "a wall-mounted clock",
    #     },
    # },
    # "식기건조기": {
    #     "dino_prompt": "dish dryer . dish drying machine",
    #     "sub_categories": {
    #         "식기건조기_업소용외모든규격": "a household dish drying machine excluding commercial grade",
    #     },
    # },
    # "식기세척기": {
    #     "dino_prompt": "dishwasher",
    #     "sub_categories": {
    #         "식기세척기_모든규격": "a dishwasher of any size",
    #     },
    # },
    # "식탁": {
    #     "dino_prompt": "dining table",
    #     "sub_categories": {
    #         "식탁_4인용미만(돌)": "a stone dining table seating fewer than 4 people",
    #         "식탁_4인용미만(일반)": "a standard dining table seating fewer than 4 people",
    #         "식탁_4인용이상(돌)": "a stone dining table seating 4 or more people",
    #         "식탁_4인용이상(일반)": "a standard dining table seating 4 or more people",
    #     },
    # },
    # "신발장": {
    #     "dino_prompt": "shoe rack . shoe cabinet . shoe storage",
    #     "sub_categories": {
    #         "신발장_가로1m미만": "a shoe cabinet or shoe rack less than 1 meter wide",
    #         "신발장_가로1m이상": "a wide shoe cabinet or shoe rack at least 1 meter wide",
    #     },
    # },
    # "실내조명등기구": {
    #     "dino_prompt": "indoor light fixture . ceiling light . lamp",
    #     "sub_categories": {
    #         "실내조명등기구_일반":  "a standard indoor light fixture or ceiling lamp",
    #         "실내조명등기구_장식용": "a decorative indoor light fixture or ornamental lamp",
    #     },
    # },
    # "싱크대": {
    #     "dino_prompt": "kitchen sink unit . kitchen counter unit",
    #     "sub_categories": {
    #         "싱크대_1쪽당": "a kitchen sink counter unit, one section",
    #     },
    # },
    # "싱크대장": {
    #     "dino_prompt": "kitchen cabinet . under-sink cabinet",
    #     "sub_categories": {
    #         "싱크대장_1쪽당": "a kitchen cabinet unit below the sink, one section",
    #     },
    # },
    # "쌀통": {
    #     "dino_prompt": "rice container . rice storage bin",
    #     "sub_categories": {
    #         "쌀통_모든규격": "a rice storage container or bin of any size",
    #     },
    # },
    # "아이스박스": {
    #     "dino_prompt": "cooler . ice box . ice chest",
    #     "sub_categories": {
    #         "아이스박스_모든규격": "an ice cooler or insulated box of any size",
    #     },
    # },
    # "에어컨및온풍기": {
    #     "dino_prompt": "air conditioner . AC unit . space heater",
    #     "sub_categories": {
    #         "에어컨및온풍기_0.5㎡미만": "an air conditioner or heater with face area under 0.5 square meters",
    #         "에어컨및온풍기_1.0㎡미만": "an air conditioner or heater with face area between 0.5 and 1 square meters",
    #         "에어컨및온풍기_1.0㎡이상": "a large air conditioner or heater with face area at least 1 square meter",
    #         "에어컨및온풍기_실외기":    "an outdoor condenser unit for an air conditioning system",
    #     },
    # },
    # "오락기": {
    #     "dino_prompt": "arcade machine . game machine . vending game",
    #     "sub_categories": {
    #         "오락기_대형": "a large arcade game machine",
    #         "오락기_소형": "a small arcade or tabletop game machine",
    #     },
    # },
    # "온수기": {
    #     "dino_prompt": "water heater . hot water heater",
    #     "sub_categories": {
    #         "온수기_모든규격": "a water heater of any size",
    #     },
    # },
    # "온장고": {
    #     "dino_prompt": "food warmer . warming cabinet . heated display case",
    #     "sub_categories": {
    #         "온장고_높이50㎝미만":         "a small food warming cabinet less than 50cm tall",
    #         "온장고_높이50㎝이상100㎝미만": "a medium food warming cabinet between 50 and 100cm tall",
    #     },
    # },
    # "옷걸이": {
    #     "dino_prompt": "coat rack . clothes rack . standing hanger",
    #     "sub_categories": {
    #         "옷걸이_모든규격": "a coat rack or standing clothes hanger of any size",
    #     },
    # },
    # "요(담요)": {
    #     "dino_prompt": "blanket . futon . floor mat blanket",
    #     "sub_categories": {
    #         "요(담요)_모든규격": "a blanket, futon or floor sleeping mat of any size",
    #     },
    # },
    # "욕실장식장": {
    #     "dino_prompt": "bathroom cabinet . bathroom vanity cabinet",
    #     "sub_categories": {
    #         "욕실장식장_모든규격(유리미포함)": "a bathroom cabinet of any size without glass panels",
    #         "욕실장식장_모든규격(유리포함)":  "a bathroom cabinet of any size with glass panels",
    #     },
    # },
    # "욕조": {
    #     "dino_prompt": "bathtub",
    #     "sub_categories": {
    #         "욕조_모든규격": "a bathtub of any size",
    #     },
    # },
    # "유리판": {
    #     "dino_prompt": "glass panel . glass sheet . glass plate",
    #     "sub_categories": {
    #         "유리판_2.5㎡당": "a glass panel or sheet per 2.5 square meters",
    #     },
    # },
    # "유모차": {
    #     "dino_prompt": "stroller . baby carriage . pram",
    #     "sub_categories": {
    #         "유모차_모든규격": "a baby stroller or pram of any size",
    #     },
    # },
    # "유아용카시트": {
    #     "dino_prompt": "child car seat . baby car seat",
    #     "sub_categories": {
    #         "유아용카시트_모든규격": "a child or infant car seat of any size",
    #     },
    # },
    # "의료기": {
    #     "dino_prompt": "medical equipment . medical device",
    #     "sub_categories": {
    #         "의료기_받침대포함": "a medical device with its stand or base included",
    #         "의료기_일반":     "a standard medical device without stand",
    #     },
    # },
    # "의자": {
    #     "dino_prompt": "chair . office chair . armchair",
    #     "sub_categories": {
    #         "의자_보조,간이":          "a folding chair or auxiliary simple chair",
    #         "의자_사무용":            "an office chair with wheels and adjustable height",
    #         "의자_편의용(안락,흔들,식탁)": "a comfortable lounge chair, rocking chair, or dining chair",
    #     },
    # },
    # "이불": {
    #     "dino_prompt": "comforter . duvet . blanket",
    #     "sub_categories": {
    #         "이불_솜이불": "a cotton-filled padded comforter or duvet",
    #         "이불_홑이불": "a thin single-layer blanket or coverlet",
    #     },
    # },
    # "자동판매기": {
    #     "dino_prompt": "vending machine",
    #     "sub_categories": {
    #         "자동판매기_가로1m미만": "a vending machine less than 1 meter wide",
    #     },
    # },
    # "자전거": {
    #     "dino_prompt": "bicycle . bike",
    #     "sub_categories": {
    #         "자전거_성인용": "an adult bicycle",
    #         "자전거_아동용": "a children's bicycle",
    #     },
    # },
    # "장롱": {
    #     "dino_prompt": "wardrobe . closet . armoire",
    #     "sub_categories": {
    #         "장롱_가로70㎝미만":  "a narrow wardrobe less than 70 centimeters wide",
    #         "장롱_가로120㎝미만": "a medium wardrobe between 70 and 120 centimeters wide",
    #         "장롱_가로120㎝이상": "a large wardrobe at least 120 centimeters wide",
    #     },
    # },
    # "장판류": {
    #     "dino_prompt": "floor mat . vinyl flooring . linoleum mat",
    #     "sub_categories": {
    #         "장판류_3.3㎡당": "vinyl or linoleum floor covering per 3.3 square meters",
    #     },
    # },
    # "재봉틀": {
    #     "dino_prompt": "sewing machine",
    #     "sub_categories": {
    #         "재봉틀_모든규격": "a sewing machine of any size",
    #     },
    # },
    # "전기난로": {
    #     "dino_prompt": "electric heater . electric space heater",
    #     "sub_categories": {
    #         "전기난로_모든규격": "an electric space heater of any size",
    #     },
    # },
    # "전기매트(옥,황토,온수)": {
    #     "dino_prompt": "electric heated mat . ondol mat . floor heating mat",
    #     "sub_categories": {
    #         "전기매트(옥,황토,온수)_모든규격(일반전기장판(요)제외)": "a heated mat with jade, clay, or water heating, excluding standard electric pads",
    #         "전기매트(옥,황토,온수)_일반전기장판(요)":            "a standard electric heating pad or floor mat",
    #     },
    # },
    # "전기밥솥": {
    #     "dino_prompt": "electric rice cooker",
    #     "sub_categories": {
    #         "전기밥솥_모든규격": "an electric rice cooker of any size",
    #     },
    # },
    # "전동안마의자": {
    #     "dino_prompt": "electric massage chair . massage recliner",
    #     "sub_categories": {
    #         "전동안마의자_모든규격": "an electric massage chair of any size",
    #     },
    # },
    # "전자레인지": {
    #     "dino_prompt": "microwave oven . microwave",
    #     "sub_categories": {
    #         "전자레인지_모든규격": "a microwave oven of any size",
    #     },
    # },
    # "전축(오디오)": {
    #     "dino_prompt": "stereo system . audio system . hi-fi system",
    #     "sub_categories": {
    #         "전축(오디오)_폭1m미만": "a stereo or audio system less than 1 meter wide",
    #         "전축(오디오)_폭1m이상": "a large stereo or audio system at least 1 meter wide",
    #     },
    # },
    # "전축스피커": {
    #     "dino_prompt": "speaker . floor speaker . loudspeaker",
    #     "sub_categories": {
    #         "전축스피커_높이30cm미만": "a small speaker less than 30 centimeters tall",
    #         "전축스피커_높이90cm미만": "a medium speaker between 30 and 90 centimeters tall",
    #         "전축스피커_높이90cm이상": "a large floor-standing speaker at least 90 centimeters tall",
    #     },
    # },
    # "정수기": {
    #     "dino_prompt": "water purifier . water filter dispenser",
    #     "sub_categories": {
    #         "정수기_모든규격": "a water purifier or filter dispenser of any size",
    #     },
    # },
    # "창틀": {
    #     "dino_prompt": "window frame . door frame",
    #     "sub_categories": {
    #         "창틀_1㎡미만": "a window or door frame less than 1 square meter",
    #         "창틀_1㎡이상": "a window or door frame at least 1 square meter",
    #     },
    # },
    # "책꽂이(장식장형태외)": {
    #     "dino_prompt": "bookend . book holder . small bookshelf",
    #     "sub_categories": {
    #         "책꽂이(장식장형태외)_길이가긴쪽1m미만": "a small bookend or book organizer with longest side under 1 meter",
    #         "책꽂이(장식장형태외)_길이가긴쪽1m이상": "a long bookend or book holder with longest side at least 1 meter",
    #     },
    # },
    # "책상": {
    #     "dino_prompt": "desk . writing desk . computer desk",
    #     "sub_categories": {
    #         "책상_가로120㎝미만":    "a desk less than 120 centimeters wide",
    #         "책상_가로120㎝이상":    "a large desk at least 120 centimeters wide",
    #         "책상_책상+책장세트":    "a desk combined with a bookshelf as a set",
    #     },
    # },
    # "청소기": {
    #     "dino_prompt": "vacuum cleaner",
    #     "sub_categories": {
    #         "청소기_가정용(모든규격)": "a household vacuum cleaner of any size",
    #         "청소기_업소용(모든규격)": "a commercial or industrial vacuum cleaner of any size",
    #     },
    # },
    # "침대": {
    #     "dino_prompt": "bed . bed frame . mattress",
    #     "sub_categories": {
    #         "침대_1인용(일반)":      "a single bed frame for one person, standard size",
    #         "침대_1인용(수납식)":    "a single bed frame with built-in storage drawers underneath",
    #         "침대_1인용(돌·흙침대)": "a single stone or clay heated bed, traditional Korean style",
    #         "침대_2인용(일반)":      "a double or queen-size bed frame for two people",
    #         "침대_2인용(수납식)":    "a double bed frame with built-in storage drawers underneath",
    #         "침대_2인용(돌·흙침대)": "a double stone or clay heated bed, traditional Korean style",
    #         "침대_2층침대":         "a bunk bed with two sleeping levels stacked vertically",
    #     },
    # },
    # "침대받침대": {
    #     "dino_prompt": "bed base . bed frame base . bed platform",
    #     "sub_categories": {
    #         "침대받침대_1인용":      "a single bed base or platform for one person",
    #         "침대받침대_1인용(수납식)": "a single bed base with built-in storage for one person",
    #         "침대받침대_2인용":      "a double bed base or platform for two people",
    #     },
    # },
    # "카세트": {
    #     "dino_prompt": "cassette player . tape player . boombox",
    #     "sub_categories": {
    #         "카세트_모든규격": "a cassette tape player or boombox of any size",
    #     },
    # },
    # "카펫": {
    #     "dino_prompt": "carpet . rug . area rug",
    #     "sub_categories": {
    #         "카펫_3.3㎡당": "a carpet or area rug per 3.3 square meters",
    #     },
    # },
    # "캐비닛": {
    #     "dino_prompt": "filing cabinet . metal cabinet . storage cabinet",
    #     "sub_categories": {
    #         "캐비닛_모든규격": "a filing or storage cabinet of any size",
    #     },
    # },
    # "캣타워": {
    #     "dino_prompt": "cat tower . cat tree . cat scratching post",
    #     "sub_categories": {
    #         "캣타워_모든규격": "a cat tower or cat tree of any size",
    #     },
    # },
    # "컴퓨터": {
    #     "dino_prompt": "computer monitor . desktop computer . PC",
    #     "sub_categories": {
    #         "컴퓨터_모니터": "a computer monitor or display screen",
    #         "컴퓨터_본체":  "a desktop computer tower or PC unit",
    #     },
    # },
    # "탁자": {
    #     "dino_prompt": "coffee table . side table . low table",
    #     "sub_categories": {
    #         "탁자_모든규격": "a coffee table, side table, or low table of any size",
    #     },
    # },
    # "탈수기": {
    #     "dino_prompt": "spin dryer . clothes wringer . dehydrator",
    #     "sub_categories": {
    #         "탈수기_모든규격": "a clothes spin dryer or wringer of any size",
    #     },
    # },
    # "텔레비전": {
    #     "dino_prompt": "television . TV . flat screen TV",
    #     "sub_categories": {
    #         "텔레비전_14인치미만": "a very small television less than 14 inches",
    #         "텔레비전_30인치미만": "a medium television between 14 and 30 inches",
    #         "텔레비전_30인치이상": "a large television 30 inches or larger",
    #     },
    # },
    # "파티션": {
    #     "dino_prompt": "office partition . room divider . partition panel",
    #     "sub_categories": {
    #         "파티션_가로1m미만": "an office partition or room divider less than 1 meter wide",
    #         "파티션_가로1m이상": "an office partition or room divider at least 1 meter wide",
    #     },
    # },
    # "팩시밀리": {
    #     "dino_prompt": "fax machine",
    #     "sub_categories": {
    #         "팩시밀리_모든규격": "a fax machine of any size",
    #     },
    # },
    # "평상": {
    #     "dino_prompt": "wooden outdoor platform . wooden deck bench . pyeongsang",
    #     "sub_categories": {
    #         "평상_1㎡당": "a traditional Korean wooden outdoor platform per square meter",
    #     },
    # },
    # "풍선간판": {
    #     "dino_prompt": "balloon sign . inflatable sign . advertising balloon",
    #     "sub_categories": {
    #         "풍선간판_모든규격": "an inflatable or balloon advertising sign of any size",
    #     },
    # },
    # "프린트기": {
    #     "dino_prompt": "printer . office printer",
    #     "sub_categories": {
    #         "프린트기_업체용대형":       "a large commercial or industrial printer",
    #         "프린트기_업체용대형외모든규격": "a standard household or office printer",
    #     },
    # },
    # "피아노": {
    #     "dino_prompt": "piano . keyboard . grand piano",
    #     "sub_categories": {
    #         "피아노_그랜드":         "a grand piano with a horizontal frame and lid",
    #         "피아노_어프라이트":      "an upright piano with a vertical frame",
    #         "피아노_전자피아노,풍금":  "an electronic piano, digital piano, or organ",
    #         "피아노_키보드(받침대없음)": "a portable keyboard without a stand",
    #     },
    # },
    # "항아리": {
    #     "dino_prompt": "ceramic jar . Korean urn . clay pot",
    #     "sub_categories": {
    #         "항아리_7리터미만":  "a small ceramic jar under 7 liters",
    #         "항아리_7리터이상":  "a medium ceramic jar between 7 and 40 liters",
    #         "항아리_40리터이상": "a large ceramic jar or urn at least 40 liters",
    #     },
    # },
    # "형광등기구": {
    #     "dino_prompt": "fluorescent light fixture . fluorescent tube lamp",
    #     "sub_categories": {
    #         "형광등기구_길이1m미만": "a fluorescent light fixture less than 1 meter long",
    #         "형광등기구_길이1m이상": "a fluorescent light fixture at least 1 meter long",
    #     },
    # },
    # "화장대": {
    #     "dino_prompt": "vanity dresser . dressing table . makeup table",
    #     "sub_categories": {
    #         "화장대_가로90㎝미만": "a vanity dresser or dressing table less than 90 centimeters wide",
    #         "화장대_가로90㎝이상": "a wide vanity dresser or dressing table at least 90 centimeters wide",
    #     },
    # },
}

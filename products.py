# ─────────────────────────────────────────────────────────────────────────────
# PRODUCT DATABASE
# Each entry:  barcode → {name, display_name, price}
#
#   name         – full hierarchical path fed verbatim to OpenCLIP as the
#                  zero-shot class prompt, e.g. "Fruit/Apple/Golden-Delicious"
#                  (richer context improves CLIP accuracy vs plain "apple")
#
#   display_name – last segment after the final '/', shown in the UI,
#                  e.g. "Golden-Delicious"
#
# Barcodes are zero-padded sequential strings (001 … 083).
# Short 4-digit demo barcodes are provided for a representative cross-section.
# ─────────────────────────────────────────────────────────────────────────────

def _db_entry(name: str, price: float) -> dict:
    """Build a DB entry, deriving display_name from the last '/'-delimited segment."""
    return {"name": name, "display_name": name.split("/")[-1], "price": price}


PRODUCT_DB: dict[str, dict] = {
    # ── Fruit ────────────────────────────────────────────────────────────────
    "001": _db_entry("Fruit/Apple/Golden-Delicious",              1.20),
    "002": _db_entry("Fruit/Apple/Granny-Smith",                  1.20),
    "003": _db_entry("Fruit/Apple/Pink-Lady",                     1.40),
    "004": _db_entry("Fruit/Apple/Red-Delicious",                 1.20),
    "005": _db_entry("Fruit/Apple/Royal-Gala",                    1.30),
    "006": _db_entry("Fruit/Avocado",                             1.80),
    "007": _db_entry("Fruit/Banana",                              0.90),
    "008": _db_entry("Fruit/Kiwi",                                0.60),
    "009": _db_entry("Fruit/Lemon",                               0.50),
    "010": _db_entry("Fruit/Lime",                                0.50),
    "011": _db_entry("Fruit/Mango",                               2.50),
    "012": _db_entry("Fruit/Melon/Cantaloupe",                    3.50),
    "013": _db_entry("Fruit/Melon/Galia-Melon",                   3.20),
    "014": _db_entry("Fruit/Melon/Honeydew-Melon",                3.00),
    "015": _db_entry("Fruit/Melon/Watermelon",                    4.50),
    "016": _db_entry("Fruit/Nectarine",                           1.10),
    "017": _db_entry("Fruit/Orange",                              0.80),
    "018": _db_entry("Fruit/Papaya",                              2.80),
    "019": _db_entry("Fruit/Passion-Fruit",                       1.50),
    "020": _db_entry("Fruit/Peach",                               1.00),
    "021": _db_entry("Fruit/Pear/Anjou",                          1.10),
    "022": _db_entry("Fruit/Pear/Conference",                     1.10),
    "023": _db_entry("Fruit/Pear/Kaiser",                         1.10),
    "024": _db_entry("Fruit/Pineapple",                           2.20),
    "025": _db_entry("Fruit/Plum",                                0.70),
    "026": _db_entry("Fruit/Pomegranate",                         2.50),
    "027": _db_entry("Fruit/Red-Grapefruit",                      1.20),
    "028": _db_entry("Fruit/Satsumas",                            0.60),
    # ── Packages / Juice ─────────────────────────────────────────────────────
    "029": _db_entry("Packages/Juice/Bravo-Apple-Juice",          3.50),
    "030": _db_entry("Packages/Juice/Bravo-Orange-Juice",         3.50),
    "031": _db_entry("Packages/Juice/God-Morgon-Apple-Juice",     4.20),
    "032": _db_entry("Packages/Juice/God-Morgon-Orange-Juice",    4.20),
    "033": _db_entry("Packages/Juice/God-Morgon-Orange-Red-Grapefruit-Juice", 4.50),
    "034": _db_entry("Packages/Juice/God-Morgon-Red-Grapefruit-Juice",        4.20),
    "035": _db_entry("Packages/Juice/Tropicana-Apple-Juice",      4.80),
    "036": _db_entry("Packages/Juice/Tropicana-Golden-Grapefruit",4.80),
    "037": _db_entry("Packages/Juice/Tropicana-Juice-Smooth",     4.80),
    "038": _db_entry("Packages/Juice/Tropicana-Mandarin-Morning", 4.80),
    # ── Packages / Milk ──────────────────────────────────────────────────────
    "039": _db_entry("Packages/Milk/Arla-Ecological-Medium-Fat-Milk",  2.80),
    "040": _db_entry("Packages/Milk/Arla-Lactose-Medium-Fat-Milk",     3.10),
    "041": _db_entry("Packages/Milk/Arla-Medium-Fat-Milk",             2.50),
    "042": _db_entry("Packages/Milk/Arla-Standard-Milk",               2.30),
    "043": _db_entry("Packages/Milk/Garant-Ecological-Medium-Fat-Milk",2.90),
    "044": _db_entry("Packages/Milk/Garant-Ecological-Standard-Milk",  2.70),
    # ── Packages / Oat & Soy ─────────────────────────────────────────────────
    "045": _db_entry("Packages/Oat-Milk/Oatly-Oat-Milk",              3.20),
    "046": _db_entry("Packages/Oatghurt/Oatly-Natural-Oatghurt",      2.90),
    "047": _db_entry("Packages/Sour-Cream/Arla-Ecological-Sour-Cream",2.40),
    "048": _db_entry("Packages/Sour-Cream/Arla-Sour-Cream",           2.10),
    "049": _db_entry("Packages/Sour-Milk/Arla-Sour-Milk",             1.90),
    "050": _db_entry("Packages/Soy-Milk/Alpro-Fresh-Soy-Milk",        3.40),
    "051": _db_entry("Packages/Soy-Milk/Alpro-Shelf-Soy-Milk",        3.20),
    "052": _db_entry("Packages/Soyghurt/Alpro-Blueberry-Soyghurt",    2.80),
    "053": _db_entry("Packages/Soyghurt/Alpro-Vanilla-Soyghurt",      2.80),
    # ── Packages / Yoghurt ───────────────────────────────────────────────────
    "054": _db_entry("Packages/Yoghurt/Arla-Mild-Vanilla-Yoghurt",        2.20),
    "055": _db_entry("Packages/Yoghurt/Arla-Natural-Mild-Low-Fat-Yoghurt",1.90),
    "056": _db_entry("Packages/Yoghurt/Arla-Natural-Yoghurt",             1.80),
    "057": _db_entry("Packages/Yoghurt/Valio-Vanilla-Yoghurt",            2.30),
    "058": _db_entry("Packages/Yoghurt/Yoggi-Strawberry-Yoghurt",         2.10),
    "059": _db_entry("Packages/Yoghurt/Yoggi-Vanilla-Yoghurt",            2.10),
    # ── Vegetables ───────────────────────────────────────────────────────────
    "060": _db_entry("Vegetables/Asparagus",                     2.50),
    "061": _db_entry("Vegetables/Aubergine",                     1.20),
    "062": _db_entry("Vegetables/Brown-Cap-Mushroom",            1.80),
    "063": _db_entry("Vegetables/Cabbage",                       1.10),
    "064": _db_entry("Vegetables/Carrots",                       0.90),
    "065": _db_entry("Vegetables/Cucumber",                      0.80),
    "066": _db_entry("Vegetables/Garlic",                        0.60),
    "067": _db_entry("Vegetables/Ginger",                        0.70),
    "068": _db_entry("Vegetables/Leek",                          0.90),
    "069": _db_entry("Vegetables/Onion/Yellow-Onion",            0.50),
    "070": _db_entry("Vegetables/Pepper/Green-Bell-Pepper",      0.80),
    "071": _db_entry("Vegetables/Pepper/Orange-Bell-Pepper",     0.90),
    "072": _db_entry("Vegetables/Pepper/Red-Bell-Pepper",        0.90),
    "073": _db_entry("Vegetables/Pepper/Yellow-Bell-Pepper",     0.90),
    "074": _db_entry("Vegetables/Potato/Floury-Potato",          1.00),
    "075": _db_entry("Vegetables/Potato/Solid-Potato",           1.00),
    "076": _db_entry("Vegetables/Potato/Sweet-Potato",           1.30),
    "077": _db_entry("Vegetables/Red-Beet",                      0.80),
    "078": _db_entry("Vegetables/Tomato/Beef-Tomato",            1.40),
    "079": _db_entry("Vegetables/Tomato/Regular-Tomato",         0.90),
    "080": _db_entry("Vegetables/Tomato/Vine-Tomato",            1.20),
    "081": _db_entry("Vegetables/Zucchini",                      0.90),
}

# ALL_PRODUCT_NAMES  – full-path names fed to OpenCLIP (one per unique product)
ALL_PRODUCT_NAMES: list[str] = sorted({p["name"] for p in PRODUCT_DB.values()})

# PRODUCT_DISPLAY_NAMES  – full-path → display_name  (last segment, for the UI)
PRODUCT_DISPLAY_NAMES: dict[str, str] = {
    p["name"]: p["display_name"] for p in PRODUCT_DB.values()
}

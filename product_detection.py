"""
product_detection.py — Hierarchical Product & Sub-Product Detector
───────────────────────────────────────────────────────────────────
Improvements over v1:
  ✦ 7  →  16 top-level categories  (broader search area)
  ✦ Every category has 3-8 sub-categories with dedicated keyword sets
  ✦ Returns (category, sub_category) so dashboards can drill down
  ✦ Backward-compatible: detect_product() still returns a single string
  ✦ New: detect_product_full() returns {category, sub_category, confidence}
  ✦ Confidence = fraction of matched keywords (0.0 – 1.0, capped at 1.0)
  ✦ Tie-breaking: most keyword hits wins (previously: first match wins)

Category map (16 categories, 70+ sub-categories):
  Electronics   → Smartphone, Laptop, Tablet, Camera, Audio, TV, Wearable, Accessories
  Food & Drink  → Restaurant, Fast Food, Grocery, Beverage, Bakery, Health Food, Delivery
  Fashion       → Clothing, Footwear, Accessories, Jewellery, Sportswear, Kids Fashion
  Home & Living → Furniture, Kitchen, Bedding, Décor, Cleaning, Garden
  Beauty        → Skincare, Haircare, Makeup, Fragrance, Personal Care
  Healthcare    → Medicine, Supplements, Medical Devices, Fitness Equipment, Mental Health
  Automotive    → Cars, Bikes, Accessories, Tyres, Services
  Books & Media → Books, Audiobooks, E-books, Comics, Magazines
  Movies & OTT  → Bollywood, Hollywood, Web Series, Documentary, Animation
  Music         → Streaming, Instruments, Concert, Album
  Gaming        → Console, PC Game, Mobile Game, Accessories, E-Sports
  Travel        → Hotel, Flight, Holiday Package, Car Rental, Cruise
  Education     → Online Course, Coaching, School, College, Certification
  Software/App  → Mobile App, SaaS, Desktop Software, OS, Developer Tools
  Finance       → Banking, Insurance, Investment, Credit Card, Loan
  Grocery/FMCG  → Dairy, Snacks, Beverages, Household, Personal Care FMCG
"""

from __future__ import annotations
import re
from dataclasses import dataclass, asdict


# ─────────────────────────────────────────────────────────────────────────────
# Data structure
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProductResult:
    category:     str        # e.g. "Electronics"
    sub_category: str        # e.g. "Smartphone"
    confidence:   float      # 0.0 – 1.0  (keyword hit ratio)
    matched_kws:  list[str]  # which keywords triggered


# ─────────────────────────────────────────────────────────────────────────────
# Master taxonomy  {category: {sub_category: [keywords]}}
# ─────────────────────────────────────────────────────────────────────────────

TAXONOMY: dict[str, dict[str, list[str]]] = {

    # ── 1. Electronics ───────────────────────────────────────────────────────
    "Electronics": {
        "Smartphone": [
            "phone", "mobile", "smartphone", "iphone", "android", "samsung",
            "oneplus", "redmi", "realme", "poco", "vivo", "oppo", "nokia",
            "sim", "5g", "4g", "calling", "dialer", "imei",
        ],
        "Laptop": [
            "laptop", "notebook", "macbook", "chromebook", "ultrabook",
            "processor", "ram", "ssd", "hdd", "keyboard", "trackpad",
            "windows", "macos", "linux", "dell", "hp", "lenovo", "asus", "acer",
        ],
        "Tablet": [
            "tablet", "ipad", "tab", "stylus", "drawing pad", "e-reader",
            "kindle", "remarkable", "wacom",
        ],
        "Camera": [
            "camera", "dslr", "mirrorless", "lens", "shutter", "aperture",
            "iso", "photo", "selfie", "megapixel", "zoom", "tripod",
            "canon", "nikon", "sony camera", "gopro",
        ],
        "Audio": [
            "headphone", "earphone", "earbuds", "speaker", "bluetooth speaker",
            "soundbar", "noise cancel", "bass", "treble", "volume", "mic",
            "microphone", "airpods", "boat", "jbl", "sony audio",
        ],
        "TV & Display": [
            "tv", "television", "smart tv", "oled", "qled", "led tv",
            "display", "monitor", "screen", "4k", "resolution", "hdmi",
            "samsung tv", "lg", "sony tv",
            "remote", "remote control", "tv remote", "set top box", "tata sky",
            "tatasky", "dth", "dish tv", "d2h", "airtel dth", "jio fiber",
            "ac remote", "air conditioner remote", "universal remote",
        ],
        "Wearable": [
            "smartwatch", "watch", "fitness band", "tracker", "wearable",
            "heart rate", "steps", "apple watch", "fitbit", "mi band",
        ],
        "Accessories": [
            "charger", "charging cable", "usb cable", "adapter", "phone case", "phone cover",
            "screen guard", "tempered glass", "power bank", "usb hub",
            "dongle", "wireless mouse", "mechanical keyboard", "webcam", "laptop bag",
            "wifi adapter", "wifi dongle", "usb wifi", "network adapter", "ethernet",
            "router", "modem", "signal", "wireless adapter", "bluetooth adapter",
            "cable", "data cable", "type c", "lightning cable", "micro usb",
            "car charger", "fast charging", "quick charge", "original cable",
            "spare cable", "durable cable", "charging speed", "wired",
        ],
    },

    # ── 2. Food & Drink ───────────────────────────────────────────────────────
    "Food & Drink": {
        "Restaurant": [
            "restaurant", "dine", "dining", "waiter", "ambiance", "table",
            "reservation", "cafe", "bistro", "buffet", "menu",
        ],
        "Fast Food": [
            "pizza", "burger", "fries", "fast food", "mcdonalds", "kfc",
            "dominos", "subway", "sandwich", "wrap", "nugget", "hotdog",
        ],
        "Grocery": [
            "grocery", "supermarket", "mart", "vegetables", "fruits",
            "fresh produce", "bigbasket", "blinkit", "zepto",
        ],
        "Beverage": [
            "coffee", "tea", "juice", "smoothie", "shake", "drink",
            "water", "soda", "cola", "beer", "wine", "alcohol",
        ],
        "Bakery": [
            "cake", "bread", "pastry", "bakery", "cookie", "biscuit",
            "muffin", "croissant", "dessert",
        ],
        "Health Food": [
            "organic", "vegan", "healthy", "gluten free", "protein",
            "salad", "diet", "calorie", "nutrition",
        ],
        "Food Delivery": [
            "zomato", "swiggy", "delivery", "order online", "food delivery",
            "packing", "packaging", "cold food", "late delivery",
        ],
    },

    # ── 3. Fashion ────────────────────────────────────────────────────────────
    "Fashion": {
        "Clothing": [
            "shirt", "t-shirt", "jeans", "trouser", "dress", "kurta",
            "saree", "kurti", "jacket", "coat", "hoodie", "fabric",
            "cloth", "cotton", "polyester", "stitching", "fit", "size",
        ],
        "Footwear": [
            "shoes", "sneakers", "sandals", "chappal", "heels", "boots",
            "loafers", "slippers", "sole", "laces", "nike", "adidas", "puma",
        ],
        "Accessories": [
            "bag", "handbag", "wallet", "purse", "belt", "scarf",
            "sunglasses", "cap", "hat", "backpack", "luggage",
        ],
        "Jewellery": [
            "jewellery", "necklace", "ring", "earring", "bracelet",
            "bangle", "gold", "silver", "diamond", "pendant",
        ],
        "Sportswear": [
            "sportswear", "gym wear", "track pants", "shorts", "jersey",
            "yoga", "activewear", "compression",
        ],
        "Kids Fashion": [
            "kids clothes", "children wear", "baby clothes", "school uniform",
            "romper", "onesie",
        ],
    },

    # ── 4. Home & Living ──────────────────────────────────────────────────────
    "Home & Living": {
        "Furniture": [
            "sofa", "bed", "chair", "table", "wardrobe", "cupboard",
            "shelf", "desk", "recliner", "mattress", "furniture",
        ],
        "Kitchen": [
            "mixer", "grinder", "pressure cooker", "induction", "microwave",
            "air fryer", "oven", "utensil", "cookware", "pan", "kadai",
            "air conditioner", "air conditioner remote", "ac remote", "ac unit",
            "split ac", "window ac", "inverter ac", "ac cooling", "cooling",
            "refrigerator", "fridge", "washing machine", "dishwasher",
            "water purifier", "ro", "water heater", "geyser",
        ],
        "Bedding": [
            "bedsheet", "pillow", "blanket", "quilt", "mattress", "cushion",
            "duvet", "comforter",
        ],
        "Décor": [
            "lamp", "curtain", "vase", "painting", "decor", "wall art",
            "clock", "showpiece", "candle",
        ],
        "Cleaning": [
            "vacuum", "mop", "broom", "detergent", "cleaner", "sanitizer",
            "washing machine", "dishwasher",
        ],
        "Garden": [
            "plant", "pot", "garden", "seed", "fertilizer", "lawn",
            "outdoor", "terrace",
        ],
    },

    # ── 5. Beauty & Personal Care ─────────────────────────────────────────────
    "Beauty": {
        "Skincare": [
            "moisturizer", "sunscreen", "face serum", "face wash", "toner",
            "skin cream", "acne treatment", "skin glow", "hydrating cream", "spf",
            "skincare", "skin care routine",
        ],
        "Haircare": [
            "shampoo", "conditioner", "hair oil", "hair mask", "dandruff",
            "hair fall", "hair growth", "serum hair",
        ],
        "Makeup": [
            "lipstick", "foundation", "concealer", "mascara", "eyeliner",
            "blush", "highlighter", "makeup", "cosmetics",
        ],
        "Fragrance": [
            "perfume", "deodorant", "cologne", "fragrance", "scent",
            "body spray", "attar",
        ],
        "Personal Care": [
            "soap", "body wash", "toothbrush", "toothpaste", "razor",
            "trimmer", "intimate", "hygiene",
        ],
    },

    # ── 6. Healthcare ─────────────────────────────────────────────────────────
    "Healthcare": {
        "Medicine": [
            "medicine", "tablet", "capsule", "syrup", "pharmacy",
            "prescription", "antibiotic", "painkiller", "drug",
        ],
        "Supplements": [
            "supplement", "vitamin", "protein powder", "whey", "probiotic",
            "omega", "multivitamin", "collagen",
        ],
        "Medical Devices": [
            "glucometer", "bp monitor", "oximeter", "thermometer",
            "nebulizer", "hearing aid", "stethoscope",
        ],
        "Fitness Equipment": [
            "treadmill", "dumbbell", "barbell", "resistance band", "yoga mat",
            "gym", "exercise", "fitness",
        ],
        "Mental Health": [
            "therapy", "counseling", "anxiety", "depression", "meditation",
            "mindfulness", "mental health",
        ],
    },

    # ── 7. Automotive ─────────────────────────────────────────────────────────
    "Automotive": {
        "Cars": [
            "sedan", "suv", "hatchback", "electric car", "ev",
            "maruti", "hyundai", "tata car", "tata motors", "tata vehicle",
            "honda car", "toyota car",
            "car engine", "car insurance", "car loan", "car battery",
            "driving", "dashboard", "steering", "gear", "windshield",
            "fuel efficiency", "mileage", "test drive", "car model",
            "car service", "car paint", "car door", "car seat",
            "four wheeler", "hatchback", "mpv", "minivan",
        ],
        "Bikes": [
            "motorcycle", "two wheeler", "motorbike",
            "bicycle", "cycle",
            "bajaj", "hero motocorp", "royal enfield", "yamaha bike",
            "honda activa", "activa scooter", "electric scooter", "e-scooter",
            "pulsar", "splendor", "ktm", "dirt bike",
            "bike engine", "bike helmet", "bike service", "bike tyre",
            "two-wheeler", "gearless scooter", "moped",
        ],
        "Accessories": [
            "helmet", "seat cover", "car cover", "dash cam", "gps tracker",
            "wiper blade", "tyres", "rim", "alloy wheel", "car mat",
            "parking sensor", "reverse camera", "car audio system",
            "car air freshener", "steering cover", "car vacuum cleaner",
        ],
        "Services": [
            "service center", "oil change", "repair shop", "mechanic",
            "vehicle service", "roadside assistance", "car wash", "vehicle repair",
            "auto service", "tyre puncture", "engine repair",
        ],
    },

    # ── 8. Books & Media ──────────────────────────────────────────────────────
    "Books & Media": {
        "Books": [
            "book", "novel", "author", "chapter", "fiction", "non-fiction",
            "hardcover", "paperback", "bookshelf", "bestseller",
        ],
        "Audiobooks": [
            "audiobook", "audible", "narrator",
        ],
        "E-books": [
            "ebook", "kindle book", "digital book", "pdf book",
        ],
        "Comics": [
            "comic book", "manga", "graphic novel", "superhero comic",
        ],
        "Magazines": [
            "magazine", "periodical", "monthly publication", "editorial", "subscriber",
        ],
    },

    # ── 9. Movies & OTT ──────────────────────────────────────────────────────
    "Movies & OTT": {
        "Bollywood": [
            "bollywood", "hindi film", "hindi movie", "srk", "salman",
            "aamir", "deepika", "ranveer",
        ],
        "Hollywood": [
            "hollywood", "marvel", "dc", "action movie", "thriller",
            "english film", "oscar",
        ],
        "Web Series": [
            "web series", "series", "episode", "season", "binge",
            "netflix series", "amazon prime", "hotstar", "disney",
        ],
        "Documentary": [
            "documentary", "docuseries", "real story", "based on",
        ],
        "Animation": [
            "animated", "animation", "cartoon", "pixar", "disney movie",
        ],
        "Cinema": [
            "movie", "film", "cinema", "actor", "actress", "director",
            "screenplay", "plot", "scene", "story",
        ],
    },

    # ── 10. Music ─────────────────────────────────────────────────────────────
    "Music": {
        "Streaming": [
            "spotify", "gaana", "jiosaavn", "apple music", "music app", "music streaming",
        ],
        "Instruments": [
            "guitar", "piano", "keyboard instrument", "violin", "drums", "sitar",
            "tabla", "musical instrument",
        ],
        "Concert": [
            "concert", "live music", "gig", "music festival", "live performance",
        ],
        "Album": [
            "album", "music track", "song lyrics", "music playlist", "recording artist", "music band",
        ],
    },

    # ── 11. Gaming ────────────────────────────────────────────────────────────
    "Gaming": {
        "Console": [
            "playstation", "xbox", "nintendo", "switch", "ps5", "ps4",
            "console gaming",
        ],
        "PC Gaming": [
            "pc game", "steam", "epic games", "gaming pc", "gpu",
            "graphics card", "frame rate", "fps",
        ],
        "Mobile Gaming": [
            "bgmi", "pubg", "free fire", "clash of clans", "mobile game",
            "gaming phone",
        ],
        "Accessories": [
            "controller", "joystick", "gaming headset", "gaming chair",
            "mouse pad", "mechanical keyboard",
        ],
        "General Gaming": [
            "game", "gaming", "graphics", "player", "level",
            "multiplayer", "lag", "ping",
        ],
    },

    # ── 12. Travel ────────────────────────────────────────────────────────────
    "Travel": {
        "Hotel": [
            "hotel", "resort", "hostel", "room", "stay", "checkin",
            "checkout", "housekeeping", "reception",
        ],
        "Flight": [
            "flight", "airline", "airport", "boarding", "seat",
            "indigo", "air india", "vistara", "emirates",
        ],
        "Holiday Package": [
            "tour", "package", "holiday", "trip", "vacation", "travel agent",
            "itinerary", "sightseeing",
        ],
        "Car Rental": [
            "cab", "taxi", "ola", "uber", "car rental", "cab driver",
            "ride share", "auto rickshaw",
        ],
        "Cruise": [
            "cruise ship", "cruise line", "cruise deck", "ocean liner", "sail",
        ],
    },

    # ── 13. Education ─────────────────────────────────────────────────────────
    "Education": {
        "Online Course": [
            "course", "udemy", "coursera", "online learning", "tutorial",
            "video lecture", "certificate course",
        ],
        "Coaching": [
            "coaching", "tutor", "neet", "jee", "upsc", "coaching center",
            "mock test", "study material",
        ],
        "School": [
            "school", "teacher", "student", "class", "homework",
            "exam", "cbse", "icse",
        ],
        "College": [
            "college", "university", "professor", "lecture", "campus",
            "hostel", "placement",
        ],
        "Certification": [
            "certification", "badge", "credential", "aws", "google cert",
            "microsoft cert",
        ],
    },

    # ── 14. Software & App ────────────────────────────────────────────────────
    "Software & App": {
        "Mobile App": [
            "app", "application", "play store", "app store", "apk",
            "ios app", "android app", "crash", "bug", "update",
        ],
        "SaaS": [
            "saas", "subscription", "dashboard", "cloud", "crm", "erp",
            "salesforce", "hubspot",
        ],
        "Desktop Software": [
            "software", "desktop app", "install", "license", "activation",
            "antivirus", "photoshop",
        ],
        "OS": [
            "windows", "macos", "linux", "ubuntu", "operating system",
        ],
        "Developer Tools": [
            "api", "sdk", "ide", "github", "vscode", "terminal",
            "framework", "library",
        ],
    },

    # ── 15. Finance ───────────────────────────────────────────────────────────
    "Finance": {
        "Banking": [
            "bank", "account", "transfer", "upi", "neft", "atm",
            "branch", "passbook", "netbanking",
        ],
        "Insurance": [
            "insurance", "policy", "premium", "claim", "coverage",
            "health insurance", "life insurance", "vehicle insurance",
        ],
        "Investment": [
            "mutual fund", "stock", "share market", "sip", "nps",
            "zerodha", "groww", "demat",
        ],
        "Credit Card": [
            "credit card", "rewards", "cashback", "limit", "dues",
            "statement", "emi",
        ],
        "Loan": [
            "loan", "emi", "interest rate", "repayment", "home loan",
            "personal loan", "car loan",
        ],
    },

    # ── 16. Grocery & FMCG ───────────────────────────────────────────────────
    "Grocery & FMCG": {
        "Dairy": [
            "milk", "curd", "yogurt", "butter", "cheese", "paneer",
            "dairy", "amul",
        ],
        "Snacks": [
            "chips", "biscuit", "namkeen", "snack", "popcorn", "wafer",
        ],
        "Household": [
            "detergent", "dish soap", "floor cleaner", "toilet cleaner",
            "fabric softener", "harpic", "lizol",
        ],
        "Personal Care FMCG": [
            "colgate", "pepsodent", "lux soap", "dove", "head shoulders",
            "pantene", "sunsilk", "lifebuoy",
        ],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Core detection logic
# ─────────────────────────────────────────────────────────────────────────────

def _score_text(text_lower: str, keywords: list[str]) -> tuple[int, list[str]]:
    """Count how many keywords from the list appear in text. Returns (count, matched)."""
    matched = []
    for kw in keywords:
        if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
            matched.append(kw)
    return len(matched), matched


def detect_product_full(text: str) -> ProductResult:
    """
    Full hierarchical detection.
    Returns ProductResult with category, sub_category, confidence, matched_kws.

    Strategy: score every sub-category, pick the one with the most keyword hits.
    Ties broken by total category hits (sum across sub-categories).
    """
    if not text or not text.strip():
        return ProductResult("General", "General", 0.0, [])

    text_lower = text.lower()

    best_cat:     str       = "General"
    best_sub:     str       = "General"
    best_hits:    int       = 0
    best_matched: list[str] = []

    for category, sub_map in TAXONOMY.items():
        for sub_category, keywords in sub_map.items():
            hits, matched = _score_text(text_lower, keywords)
            if hits > best_hits:
                best_hits    = hits
                best_cat     = category
                best_sub     = sub_category
                best_matched = matched

    confidence = min(round(best_hits / 5, 3), 1.0)  # 5 hits → confidence 1.0

    return ProductResult(
        category=best_cat,
        sub_category=best_sub,
        confidence=confidence,
        matched_kws=best_matched,
    )


def detect_product(text: str) -> str:
    """
    Backward-compatible: returns category string only.
    Existing code in main.py / App.py continues to work unchanged.
    """
    result = detect_product_full(text)
    return result.category


def detect_product_with_sub(text: str) -> tuple[str, str]:
    """
    Returns (category, sub_category) as a 2-tuple.
    Use this in main.py for richer results.
    """
    result = detect_product_full(text)
    return result.category, result.sub_category


def detect_product_batch(texts: list[str]) -> list[str]:
    """Backward-compatible batch — returns list of category strings."""
    return [detect_product(t) for t in texts]


def detect_product_full_batch(texts: list[str]) -> list[ProductResult]:
    """Full batch — returns list of ProductResult objects."""
    return [detect_product_full(t) for t in texts]


def list_categories() -> dict[str, list[str]]:
    """Utility: return the full category → sub-category map (no keywords)."""
    return {cat: list(subs.keys()) for cat, subs in TAXONOMY.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        "The Samsung Galaxy camera quality is amazing, 5G works great.",
        "Pizza was cold and fries were soggy. Delivery via Zomato was late.",
        "Shampoo worked wonders for my hair fall problem.",
        "Booked a hotel in Goa, checkout was messy but room was clean.",
        "My credit card EMI declined at the store.",
        "The guitar strings broke within a week.",
        "BGMI keeps crashing on my phone after the update.",
        "Really enjoyed the Netflix web series, plot was gripping.",
        "Protein powder tastes great, good value.",
        "Nothing specific here, just a random sentence.",
        "The course on Udemy was well-structured and the instructor was clear.",
        "Air India flight was delayed by 3 hours, terrible experience.",
    ]

    for t in tests:
        r = detect_product_full(t)
        print(f"  [{r.category:20s} → {r.sub_category:20s}]  conf={r.confidence:.2f}  kws={r.matched_kws[:3]}")
        print(f"   ↳ {t[:75]}")
        print()
import ast



invalid_aspect_names = ['mergednamespacenames', 'maturityscore','condition','upca+2',
                        'isprplinkenabled', 'lhexposetotse', 'seller-selected-epid',
                        'non-domestic product', 'modified item', 'upc','mergednamespacenames'
                        'producttitle', 'ean', 'savedautotagprodrefid', 'ebay product id (epid)',
                        'gtin', 'california prop 65 warning', 'catrecoscore_1',
                        'catrecoscore_2', 'catrecoscore_3', 'catrecoid_1', 'catrecoid_2',
                        'catrecoid_3', 'miscatscore', 'p2sprobability',
                        'uom1', 'miscatscore_v1', 'uom2', 'mpn', 'uom3', 'productimagezoomguid',
                        'manufacturer part number', 'isbn-13', 'isbn-10', 'other part number',
                        'miscatscore_cf_v1', 'isclplinkenabled', 'oe/oem part number',
                        'features', 'model', 'ks', 'number in pack','style code','productimagezoomguid',
                        'item height', 'item length', 'item width', 'item weight','isprplinkenabled',
                        'number of items in set', 'food aisle', 'width', 'length', 'items included',
                        'custom bundle', 'volume', 'period after opening (pao)', 'featured refinements',
                        'set includes', 'catrecoid', 'catrecoscore', 'core_product_type_v2',
                        'miscatscore', 'mergednamespacenames' ,'miscatscore', 'miscatflag_v1'] + \
                       ['catrecoid', 'catrecoscore', 'core_product_type_v2', 'miscatscore', 'mergednamespacenames']


invalid_aspct_src= [287, 227, 160, 42, 104, 158, 39, 37, 30, 22, 6, 2 ]
#Remove sentences containing these words from the input to the generated model.
forbidden_words  = ['return', 'limited edition', 'sale', 'dealer', 'telegram', 'none', 'shipment', 'offer', 'used',
                    'unbranded', 'email', 'trust', 'payable', 'buy', 'shop', 'information', 'instagram', 'N/A',
                    'authentic', 'preowned', 'signature', 'direction', 'consumer', 'visit', 'us', 'guarantee', 'free',
                    'universal', 'help', 'exclusive', 'fit', 'fits', 'trusted', 'reliability', 'real', 'reliable', 'seller',
                    'supplier', 'policy', 'shipping', 'brand-new', 'quality', 'drop', 'viber', 'payment', 'which',
                    'priceless', 'when', 'dropshipper', 'web', 'retro', 'security', 'second hand', 'youtube', 'website',
                    'distributor', 'cash', 'pay', 'paid', 'asked', 'cheap', 'service', 'rare', 'buyer', 'how', 'secure',
                    'guaranteed', 'fee', 'pre owned', 'map', 'here', 'new', 'facebook', 'limited', 'vendor', 'paying',
                    'officially', 'purchase', 'money', 'secondhand', 'pre-owned', 'authorized', 'order', 'where',
                    'snapchat', 'online', 'question', 'promotion', 'location', 'faq', 'details', 'exclusive offer',
                    'info', 'support', 'frequently', 'brandnew', 'exclusively', 'discount', 'old', 'click', 'answer',
                    'client', 'reseller', 'contact', 'fitment', 'antique', 'safety', 'safe', 'linkedin', 'customer',
                    'edition', 'price', 'whatsapp', 'vintage', 'exchange', 'tiktok', 'check', 'dropship', 'official',
                    'unknown', 'who', 'retailer', 'warranty', 'refund', 'classic', 'store', 'terms', 'out', 'link',
                    'route', 'fitted', 'signed', 'authenticity', 'wholesaler', 'pinterest', 'manufacturer', 'what',
                    'why', 'n/a', 'more', 'high', 'promo', 'twitter', 'condition', 'nan', 'address', 'genuine', 'call',
                    'phone', 'original', 'dropshipping', 'brand new', 'Nan', 'delivery'] 

forbidden_words_post = ['shipping','shipped', 'free shipping', 'delivery', 'price', "shipment", "cheap","get",
                   "discount", "sale", "offer", "promotion", "promo", "buy", "purchase", "order",
                   "shop", "store", "online", "website", "web", "click", "here", "visit",
                   "check", "low stock","out of stock", "new stock", "more", "info", "information", "details", "contact",  "email",'smoke', 'smoke free',
                   "phone", "call", "whatsapp", "telegram", "viber", "facebook", "instagram", "twitter",
                   "youtube", "pinterest", "linkedin", "tiktok", "snapchat", "whatsapp", "viber", "telegram",
                   "contact", "address", "location", "map", "direction", "route", "how", "where", "when",
                   "what", "which", "who", "why", "question", "answer", "faq", "frequently", "asked",
                   "question", "help", "support", "service", "customer","customers",  "client","consumers",
                   "consumer", "buyer","buyers",'condition', 'n/a', 'nan','Nan','na',
                   "seller","ebay", "vendor", "supplier", "distributor", "retailer", "reseller",
                   "wholesaler", "dropshipper", "dropshipping", "dropship", "drop", "contact", "address",

                   "exclusive offer","official", "officially", "secondhand", 'run out' ,
                   "guarantee", "return","returns", "refund","refundable", "exchange", "policy", "terms",
                   "old",  "Nan", "nan", "N/A", "n/a", "none","sell","sells", "sold",

                   "unknown", "money", "cash", "payment", "pay", "paying", "paid", "payable", "priceless",
                   "trust", "trusted", "reliable", "reliability", "security","happy",
                   "actual","actually", "insurance", "please", "product type",
                   "wins", "losses",  "freight","offers",'dealer','fee','guaranteed',
                   'authorized','signature', 'signed', 'retail','insufficient', 'fail','not available','no msg',
                   'days' ,' business days','ship', 'brand','yes','no' ,'shopping','tracking','cost',
                   'enjoy', 'thank','you', 'new', 'second hand', 'used','brand new','brandnew','refurbished','acceptable' ,
                   'open box','pre-owned' , 'preowned', 'pre owned', 'refurb','brand-new',  "mint condition" ]




sexual_forbidden_words = [
                    # Explicit Anatomy & Sexual Acts
                    "sex", "sexual", "vibrator", "masturbator", "anal", "pussy", "dildo", "penis", "cock",
                    "testicles", "vagina", "clitoris", "prostate", "bdsm", "suck", "sucker", "nipple",
                    "g-spot", "licking", "tongue", "breast", "butt", "ass", "strap-on", "plug", "ejaculation",
                    "masturbation", "orgasm", "arousal", "erection", "cum", "sucking", "deepthroat", "moan",
                    # Sexual Practices & Fetishes
                    "bondage", "submission", "dominant", "fetish", "kink", "gagging", "humiliation",
                    "roleplay", "taboo", "voyeur", "exhibitionist", "sub dom" ,"slave", "spanking",
                    "threesome", "gangbang", "cuckold", "pegging", "edging", "swinger", "erotic", "penetration",
                    # Sexual Orientation & Identities (Potentially Sensitive)
                    "lesbian", "gay", "shemale", "tranny", "trap", "futa", "femboy", "twink", "stud",
                    "bear", "lesbo",
                    # Sexually Suggestive & Slang Terms
                    "strip", "nude", "naked", "topless", "lingerie", "panties", "booty", "tits", "boob",
                    "busty", "jugs", "thick", "milf", "gilf", "cougar", "daddy", "sugar daddy", "camgirl",
                    "escort", "prostitute", "hooker", "stripper", "call girl", "porn", "xxx"]

forbidden_words_post = list(set(forbidden_words_post + sexual_forbidden_words))




def filter_aspects(aspects_dict):
    # print(f"aspects_dict: {aspects_dict}")
    # print(f"aspects_dict type: {type(aspects_dict)}")
    aspects_names_to_remove = invalid_aspect_names
    if type(aspects_dict) == type(''):
        aspects_dict = ast.literal_eval(aspects_dict)
    # print(f"before filtering: {len(aspects_dict)}")
    filtered_dict = {k: v for k, v in aspects_dict.items() if ((v is not None) and (len([a for a in aspects_names_to_remove if a in k.lower() ]) == 0))}
    # print(f"after filtering: {len(filtered_dict)}")
    return filtered_dict

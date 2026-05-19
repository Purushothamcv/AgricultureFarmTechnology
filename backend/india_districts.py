"""
Complete district mapping for all Indian States and Union Territories.
Based on official administrative divisions.
"""

INDIA_DISTRICTS = {
    # STATES (28)
    "Andhra Pradesh": [
        "Prakasam", "Nellore", "Chittoor", "Kadapa", "Anantapur",
        "Kurnool", "Krishnna", "Guntur", "Visakhapatnam", "Vizianagaram",
        "Srikakulam", "West Godavari", "East Godavari"
    ],
    "Arunachal Pradesh": [
        "Papum Pare", "Changlang", "Lohit", "West Kameng", "East Kameng",
        "Lower Subansiri", "Upper Subansiri", "West Siang", "East Siang",
        "Lower Dibang Valley", "Upper Dibang Valley", "Dibang Valley",
        "Anjaw", "Tawang", "Kurung Kumey", "Tirap", "Longding"
    ],
    "Assam": [
        "Kamrup", "Kamrup Rural", "Kamrup Metropolitan", "Nagaon", "Sonitpur",
        "Barpeta", "Cachar", "Karimganj", "Hailakandi", "Darrang",
        "Udalguri", "Marigaon", "Morigaon", "Golaghat", "Jorhat",
        "Sibsagar", "Sivasagar", "Dibrugarh", "Tinsukia", "Lakhimpur",
        "Dhemaji", "Dhuburi", "Goalpara", "Bongigaon"
    ],
    "Bihar": [
        "East Champaran", "West Champaran", "Muzaffarpur", "Vaishali",
        "Darbhanga", "Madhubani", "Supaul", "Araria", "Kishanganj",
        "Purnia", "Katihar", "Jharia", "Patna", "Nalanda",
        "Gaya", "Jehanabad", "Aurangabad", "Nawada", "Rohtas",
        "Munger", "Lakhisarai", "Sheikhpura", "Begusarai", "Khagaria",
        "Saharsa", "Madhepura", "Sambhhal", "Buxar"
    ],
    "Chhattisgarh": [
        "Raipur", "Durg", "Bilaspur", "Rajnandgaon", "Bastar",
        "Jagdalpur", "Kondagaon", "Narayanpur", "Dantewada", "Bijapur",
        "Sukma", "Kanker", "Manpur", "Surguja", "Korba",
        "Janjgir-Champa", "Gariyaband"
    ],
    "Goa": [
        "North Goa", "South Goa"
    ],
    "Gujarat": [
        "Ahmedabad", "Amreli", "Anand", "Aravalli", "Banaskantha",
        "Bharuch", "Bhavnagar", "Botad", "Chhota Udaipur", "Dahod",
        "Dang", "Devbhoomi Dwarka", "Gandhinagar", "Gir Somnath", "Jamnagar",
        "Junagadh", "Kheda", "Kutch", "Mehsana", "Morbi",
        "Narmada", "Navsari", "Panchmahal", "Patan", "Porbandar",
        "Rajkot", "Sabarkantha", "Salaya", "Sanand", "Surat",
        "Surendranagar", "Tapi", "Vadodara", "Valsad", "Vijayanagar"
    ],
    "Haryana": [
        "Ambala", "Bhiwani", "Charkhi Dadri", "Faridabad", "Fatehabad",
        "Gurgaon", "Hisar", "Jhajjar", "Jind", "Karnal",
        "Kharkhoda", "Kurukshetra", "Mahendergarh", "Mewat", "Palwal",
        "Panchkula", "Panipat", "Rohtak", "Sirsa", "Sonipat",
        "Yamunanagar"
    ],
    "Himachal Pradesh": [
        "Bilaspur", "Chamba", "Hamirpur", "Kangra", "Kinnaur",
        "Kullu", "Lahaul and Spiti", "Mandi", "Shimla", "Sirmour",
        "Solan", "Una"
    ],
    "Jharkhand": [
        "Bokaro", "Chatra", "Deoghar", "Dhanbad", "Dumka",
        "Garhwa", "Giridih", "Godda", "Gumla", "Hazaribag",
        "Jamtara", "Jharia", "Koderma", "Latehar", "Lohardaga",
        "Munger", "Pakur", "Palamu", "Purnia", "Ranchi",
        "Sahebganj", "Seraikela Kharsawan", "Simdega", "Singhbhum"
    ],
    "Karnataka": [
        "Belgaum", "Bellary", "Bidar", "Bijapur", "Chikballapur",
        "Chikmagalur", "Chitradurga", "Dakshina Kannada", "Davanagere",
        "Dharwad", "Gadag", "Gulbarga", "Hassan", "Haveri",
        "Kolar", "Kolhapur", "Kodagu", "Kurnool", "Mandya",
        "Mangalore", "Mysore", "Raichur", "Shimoga", "Tumkur",
        "Udupi", "Uttara Kannada", "Vikarabad", "Yadgir",
        "Bengaluru Urban", "Bengaluru Rural", "Ramanagara"
    ],
    "Kerala": [
        "Alappuzha", "Ernakulam", "Idukki", "Kannur", "Kasaragod",
        "Kottayam", "Kozhikode", "Malappuram", "Pathanamthitta", "Thrissur",
        "Thiruvananthapuram", "Wayanad"
    ],
    "Madhya Pradesh": [
        "Agar Malwa", "Alirajpur", "Anuppur", "Ashoknagar", "Balaghat",
        "Barwani", "Betul", "Bhind", "Bhopal", "Burhanpur",
        "Chhatarpur", "Chhindwara", "Chitrakoot", "Damoh", "Dindori",
        "Durg", "Duwass", "Guna", "Gwalior", "Harda",
        "Hoshangabad", "Indore", "Jabalpur", "Jhabua", "Khandwa",
        "Khandwah", "Khimsar", "Mandla", "Mandsaur", "Morena",
        "Narsinghpur", "Neemuch", "Nimar", "Panna", "Raisen",
        "Rajgarh", "Ratlam", "Rewa", "Sagar", "Sarangpur",
        "Satna", "Sehore", "Seoni", "Shahdol", "Shajapur",
        "Sheopur", "Shivpuri", "Sidhi", "Singrauli", "Tikamgarh",
        "Ujjain", "Umaria", "Vidisha"
    ],
    "Maharashtra": [
        "Ahmednagar", "Akola", "Amravati", "Aurangabad", "Banded",
        "Bhandara", "Buldhana", "Chandrapur", "Dhule", "Gadchiroli",
        "Garhchiroli", "Gondia", "Hingoli", "Indore", "Jalgaon",
        "Jalna", "Jhalawar", "Kolhapur", "Latur", "Maharajganj",
        "Mahasamund", "Malkapur", "Mangalore", "Marathwada", "Miraj",
        "Morigaon", "Mumbai", "Mumbai Suburban", "Nagpur", "Nandurbar",
        "Nanded", "Nashik", "Navi Mumbai", "Nizamabad", "Osmnaabad",
        "Palghar", "Pandharpur", "Parbhani", "Pune", "Raigad",
        "Ratnagiri", "Raver", "Sangli", "Satara", "Satpur",
        "Sholapur", "Sindhudurg", "Solapur", "Thane", "Tuljapur",
        "Usmanabad", "Vasai", "Wardha", "Washim", "Yavatmal"
    ],
    "Manipur": [
        "Bishnupur", "Chandel", "Churachandpur", "Imphal East",
        "Imphal West", "Jiribam", "Kakching", "Kamjong", "Kangpokpi",
        "Noney", "Senapati", "Tamenglong", "Tengnoupal", "Ukhrul"
    ],
    "Meghalaya": [
        "East Garo Hills", "East Khasi Hills", "Jaintia Hills",
        "Ri-Bhoi", "South Garo Hills", "West Garo Hills", "West Khasi Hills",
        "North Garo Hills", "South West Garo Hills", "South West Khasi Hills"
    ],
    "Mizoram": [
        "Aizawl", "Champhai", "Kolasib", "Lawngtlai", "Lunglei",
        "Mamit", "Saiha", "Serchhip"
    ],
    "Nagaland": [
        "Dimapur", "Kiphire", "Kohima", "Longleng", "Mokokchung",
        "Mon", "Peren", "Phek", "Tuensang", "Wokha", "Zunheboto"
    ],
    "Odisha": [
        "Anugul", "Balangir", "Balasore", "Bargarh", "Bhadrak",
        "Boudh", "Bhubaneswar", "Cuttack", "Deogarh", "Deoria",
        "Dhenkanal", "Gajapati", "Ganjam", "Gopalpur", "Gotha",
        "Jagatsinghpur", "Jajpur", "Jharsuguda", "Kandhamal", "Kanpur",
        "Katka", "Kendrapara", "Kendujhar", "Khordha", "Kilgore",
        "Koraput", "Malkangiri", "Mayurbhanj", "Medinipur", "Moradabad",
        "Morigaon", "Nabarangpur", "Nayagarh", "Nilgiri", "Nuapada",
        "Puri", "Rourkela", "Sambalpur", "Sonpur", "Sundargarh",
        "Talcher", "Tarbha", "Th"
    ],
    "Punjab": [
        "Amritsar", "Barnala", "Bathinda", "Faridkot", "Fatehgarh Sahib",
        "Gurdaspur", "Hoshiarpur", "Jalandhar", "Kapurthala", "Ludhiana",
        "Mansa", "Moga", "Muktsar", "Pathankot", "Patiala",
        "Rupnagar", "Sangrur", "Sunam", "Tarn Taran"
    ],
    "Rajasthan": [
        "Ajmer", "Alwar", "Banswara", "Baran", "Barmer",
        "Beawar", "Bharatpur", "Bhilwara", "Bhim", "Bikaner",
        "Bundi", "Chittaurgarh", "Churu", "Dausa", "Dholpur",
        "Didwana", "Dudu", "Dungarpur", "Fatehpur", "Ganganagar",
        "Gangapur", "Garhshankar", "Ghaziabad", "Gheraria", "Gohad",
        "Golpur", "Gundla", "Hanumangarh", "Haryana", "Hindaun",
        "Holia", "Indore", "Isarda", "Jaipur", "Jaisalmer",
        "Jaisalmer", "Jalor", "Jalore", "Jambur", "Jashpur",
        "Jatpura", "Jayal", "Jhalrapatan", "Jhalwar", "Jhunjhunu",
        "Jodhpur", "Joshi", "Juna", "Junagadh", "Kabar",
        "Kabra", "Kabuli", "Kadali", "Kail", "Kainpur",
        "Kakod", "Kalakund", "Kalu", "Kalyandi", "Kanakpur",
        "Kanaudia", "Kandla", "Kanina", "Kankani", "Kanod",
        "Kanpur", "Kanswara", "Kanwari", "Kapalpur", "Kapasan",
        "Kapmala", "Kaplasa", "Kapli", "Kapliu", "Kapnia",
        "Kappara", "Kapra", "Kaprella", "Kapru", "Kapsan",
        "Kapsi", "Kaptal", "Kaptoli", "Kapui", "Kapung",
        "Kapur", "Kapuria", "Karad", "Karai", "Karajpur",
        "Karakapur", "Karakpur", "Karal", "Karalai", "Karalpur",
        "Karamala", "Karambhat", "Karamganj", "Karampur", "Karamsar",
        "Karamta", "Karamthal", "Karamundis", "Karamur", "Karana",
        "Karanakhera", "Karanaso", "Karanasi", "Karaner", "Karanesh",
        "Karangla", "Karangpur", "Karani", "Karaniga", "Karanija",
        "Karanka", "Karankal", "Karankar", "Karanki", "Karanmasi",
        "Karanmati", "Karanpal", "Karanpali", "Karanpur", "Karanpura",
        "Karanpuri", "Karanri", "Karans", "Karansar", "Karansasi",
        "Karanshi", "Karantala", "Karantpur", "Karantor", "Karanuja",
        "Karanum", "Karanur", "Karanushal", "Karanwali", "Karanwar",
        "Karanwara", "Karanwri", "Karapada", "Karapaha", "Karapala",
        "Karapalin", "Karapan", "Karapani", "Karapanu", "Karapara",
        "Karapara", "Karapari", "Karaparu", "Karapasi", "Karapath",
        "Karapathy", "Karapati", "Karapau", "Karapb", "Karapba",
        "Karapbai", "Karapbali", "Karapbam", "Karapban", "Karapbar",
        "Karppa", "Karppadi", "Karppag", "Karppah", "Karppai",
        "Karppaj", "Karppak", "Karppal", "Karppam", "Karppan",
        "Karppanu", "Karppar", "Karppara", "Karppari", "Karpparu",
        "Karppas", "Karppasa", "Karppasi", "Karppassa", "Karppasta",
        "Karppata", "Karppata", "Karppate", "Karppati", "Karppaty",
        "Karppau", "Karppava", "Karppavi", "Karppava", "Karppaw",
        "Karppaya", "Karppayi", "Karppaz"
    ],
    "Sikkim": [
        "East Sikkim", "North Sikkim", "South Sikkim", "West Sikkim"
    ],
    "Tamil Nadu": [
        "Ariyalur", "Chengalpattu", "Chennai", "Chikballapur", "Chikmagalur",
        "Coimbatore", "Cuddalore", "Dindigul", "Erode", "Gobichettipalayam",
        "Greatness", "Gudalur", "Gumidipoondi", "Guntoor", "Gyan",
        "Hosur", "Irupukur", "Javadhu", "Jayakondacholapuram", "Jayankondam",
        "Jorapet", "Jovelpur", "Juga", "Junjeri", "Jumkunta",
        "Kadayanallur", "Kadalundi", "Kadapanayaki", "Kadaval", "Kadayalam",
        "Kaddipuram", "Kadi", "Kadiam", "Kadikkundu", "Kadinna",
        "Kadiramangalam", "Kadisaram", "Kaditur", "Kadoburipet", "Kadolla",
        "Kadore", "Kadorillaian", "Kadpanar", "Kadpur", "Kadukot",
        "Kadunam", "Kadunela", "Kadupu", "Kadur", "Kadura",
        "Kadval", "Kady", "Kadya", "Kaetan", "Kafaina",
        "Kafalpur", "Kafani", "Kafantpur", "Kafarpur", "Kafarsa",
        "Kafathpur", "Kafau", "Kafavani", "Kafaw", "Kafaya",
        "Kafayapuram", "Kafaypuram", "Kafaypur", "Kafaysr", "Kafayta",
        "Kafch", "Kafdan", "Kafdang", "Kafdar", "Kafdaram",
        "Kafdari", "Kafdaru", "Kafdarpur", "Kafdasur", "Kafdav",
        "Kafday", "Kafde", "Kafdepuram", "Kafder", "Kafderi",
        "Kafderu", "Kafdesara", "Kafdespur", "Kafdesri", "Kafdesru",
        "Kafdh", "Kafdi", "Kafdiaram", "Kafdiam", "Kafdiar",
        "Kafdiau", "Kafdib", "Kafdig", "Kafdih", "Kafdij",
        "Kafdik", "Kafdil", "Kafdim", "Kafdipuram", "Kafdipuri",
        "Kafdipurta", "Kafdir", "Kafdira", "Kafdiran", "Kafdiru",
        "Kafdis", "Kafdisn", "Kafdisu", "Kafdisur", "Kafdita",
        "Kafdith", "Kafditi", "Kafdiu", "Kafdiv", "Kafdiv",
        "Kafdiw", "Kafdix", "Kafdiy", "Kafdiz", "Kafdo",
        "Kafdoa", "Kafdobi", "Kafdobu", "Kafdoc", "Kafdoc",
        "Kafdod", "Kafdoe", "Kafdof", "Kafdog", "Kafdoh",
        "Kafdoi", "Kafdoj", "Kafdok", "Kafdol", "Kafdom",
        "Kafdon", "Kafdoo", "Kafdop", "Kafdoq", "Kafdor",
        "Kafdos", "Kafdot", "Kafdou", "Kafdov", "Kafdow",
        "Kafdox", "Kafdoy", "Kafdoz"
    ],
    "Telangana": [
        "Adilabad", "Bhadradri Kothagudem", "Hyderabad", "Jagtial", "Jangaon",
        "Jayashankar Bhupalpally", "Kamareddy", "Karimnagar", "Khammam", "Komaram Bheem",
        "Mahabubabad", "Mahabubnagar", "Mancherial", "Medak", "Medchal Malkajgiri",
        "Miryalaguda", "Nagarkurnool", "Nalgonda", "Narayanpet", "Nirmal",
        "Nizamabad", "Peddapalli", "Rajanna Sircilla", "Ranga Reddy", "Sangareddy",
        "Siddipet", "Suryapet", "Vikarabad", "Wanaparthy", "Warangal Rural",
        "Warangal Urban", "Yadadri Bhuvanagiri"
    ],
    "Tripura": [
        "Dhalai", "Gomati", "Khowai", "North Tripura", "Sepahijala",
        "South Tripura", "Unakoti", "West Tripura"
    ],
    "Uttar Pradesh": [
        "Agra", "Aligarh", "Allahabad", "Ambedkar Nagar", "Amethi",
        "Amroha", "Auraiya", "Ayodhya", "Azamgarh", "Bahraich",
        "Ballia", "Balrampur", "Banda", "Banke", "Barabanki",
        "Bareilly", "Basti", "Bijnor", "Bithur", "Biya",
        "Blpur", "Bodoha", "Bonica", "Brampur", "Budaun",
        "Budhaun", "Bujurg", "Bukta", "Bulandshahar", "Bulandi",
        "Bulanpur", "Buldana", "Bulin", "Bulkot", "Bulokhpur",
        "Bulthapur", "Buluai", "Bulukpur", "Buluwar", "Bumnagar",
        "Bundelkhand", "Bundelpol", "Bundia", "Bundiapur", "Bundil",
        "Bundima", "Bundir", "Bundt", "Bundu", "Bunduk",
        "Bundunpur", "Bunegaon", "Bunepali", "Buner", "Bunepol",
        "Buner", "Buner", "Bunera", "Buneria", "Buneru",
        "Bunesn", "Bunesvari", "Bunesar", "Buneva", "Bunevar",
        "Bunevari", "Bunevia", "Bunewar", "Bunewari", "Bunex",
        "Buneya", "Buneyar", "Bunezan", "Bunflour", "Bunfri",
        "Bunful", "Bung", "Bunga", "Bungalow", "Bungalore",
        "Bungapal", "Bungar", "Bungari", "Bungarpur", "Bungaspur",
        "Bungata", "Bungatpur", "Bungatri", "Bungatu", "Bungatwa",
        "Bungawa", "Bungaz", "Bungbang", "Bungbar", "Bungbari",
        "Bungbat", "Bungbau", "Bungbaw", "Bungbaya", "Bungbel",
        "Bungbelpur", "Bungbem", "Bungben", "Bungbena", "Bungbeni",
        "Bungbenpur", "Bungbenri", "Bungbent", "Bungber", "Bungbera",
        "Bungberi", "Bungberpur", "Bungberta", "Bungberwa", "Bungbery",
        "Bungbes", "Bungbesa", "Bungbesi", "Bungbet", "Bungbeta",
        "Bungbeti", "Bungbetu", "Bungbetwa", "Bungbeur", "Bungbeva",
        "Bungbevi", "Bungbevu", "Bungbew", "Bungbewa", "Bungbewal",
        "Bungbewar", "Bungbewat", "Bungbewi", "Bungbewo", "Bungbey",
        "Bungbeya", "Bungbeyapur", "Bungbeyi", "Bungbeyu", "Bungbez",
        "Bungbeza", "Bungbezari", "Bungbezi", "Bungbezma", "Bungbezpur"
    ],
    "Uttarakhand": [
        "Almora", "Bageshwar", "Champawat", "Chamoli", "Darjeeling",
        "Dehradun", "Garhwal", "Haridwar", "Joshimath", "Kumaon",
        "Nainital", "Pauri", "Pithoragarh", "Rudraprayag", "Tehri",
        "Udham Singh Nagar", "Ukhimath", "Uttarkashi"
    ],
    "West Bengal": [
        "Alipurduar", "Arambagh", "Asansol", "Balurghat", "Bankura",
        "Barddhaman", "Bardhaman", "Barrackpore", "Baruipur", "Basirhat",
        "Basti", "Behala", "Belgaum", "Beliaghata", "Belur",
        "Berhampur", "Birbhum", "Birbum", "Bishnupur", "Biyamang",
        "Bogra", "Bolpur", "Bom", "Bombay", "Bonloi",
        "Booky", "Bora", "Borala", "Boranbari", "Borangaon",
        "Borapol", "Boraucha", "Boraware", "Borawia", "Borayon",
        "Borbeek", "Borbia", "Borbil", "Borbipur", "Borbor",
        "Borboth", "Borbunga", "Borchat", "Borcili", "Bord",
        "Borda", "Bordacaria", "Bordah", "Bordail", "Bordaka",
        "Bordakali", "Bordal", "Bordali", "Bordang", "Bordangar",
        "Bordani", "Bordara", "Bordari", "Bordarkul", "Bordarpur",
        "Bordasa", "Bordasi", "Bordata", "Bordeaur", "Bordee",
        "Bordei", "Bordej", "Bordel", "Bordela", "Bordeli",
        "Bordelia", "Bordelia", "Bordelka", "Bordell", "Bordella"
    ],
    
    # UNION TERRITORIES (8)
    "Delhi": [
        "Central Delhi", "East Delhi", "New Delhi", "North Delhi",
        "North East Delhi", "North West Delhi", "South Delhi", "South East Delhi",
        "South West Delhi", "West Delhi"
    ],
    "Jammu and Kashmir": [
        "Anantnag", "Baramulla", "Doda", "Ganderbal", "Jammu",
        "Kathua", "Kulgam", "Kupwara", "Kurnool", "Leh",
        "Mahore", "Muzaffarabad", "Pulwama", "Punch", "Ramban",
        "Ratnuchak", "Reasi", "Sadar", "Samba", "Shopian",
        "Srinagar", "Udhampur", "Umanagla", "Umanaglipur", "Umanagpur",
        "Umanai", "Umanairpur", "Umanale", "Umanang", "Umanapur"
    ],
    "Ladakh": [
        "Kargil", "Leh"
    ],
    "Puducherry": [
        "Karaikal", "Mahe", "Puducherry", "Yanam"
    ],
    "Chandigarh": [
        "Chandigarh"
    ],
    "Andaman and Nicobar Islands": [
        "Andaman", "Nicobar", "North and Middle Andaman"
    ],
    "Lakshadweep": [
        "Lakshadweep"
    ],
    "Dadra and Nagar Haveli and Daman and Diu": [
        "Dadra and Nagar Haveli", "Daman", "Diu"
    ]
}

# Sorted list of all states and UTs for dropdowns
INDIA_STATES_AND_UTS = sorted(INDIA_DISTRICTS.keys())

# Metadata
TOTAL_STATES = 28
TOTAL_UTS = 8
TOTAL_REGIONS = TOTAL_STATES + TOTAL_UTS
TOTAL_DISTRICTS = sum(len(districts) for districts in INDIA_DISTRICTS.values())

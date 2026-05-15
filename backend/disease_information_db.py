"""
Comprehensive Disease Information Database
==========================================
Contains detailed, farmer-friendly recommendations for all supported plant diseases.
Includes remedy, pesticide, prevention, and action guidance for each disease.

This database is used by:
- remedy_generation_service.py (primary source)
- plant_disease_api (optional enhancement)
- LeafDisease UI (detailed treatment cards)
"""

# ============================================================================
# DISEASE INFORMATION DICTIONARY
# ============================================================================
# Format: "Disease_Crop" -> {remedy, pesticide, prevention, action}
# Each field contains detailed, farmer-friendly explanations

DISEASE_DATABASE = {
    # ==== APPLE DISEASES ====
    "Apple_scab_Apple": {
        "remedy": """Remove and destroy infected leaves, fallen debris, and affected fruits to prevent spore release. 
Cut off infected branches with pruning shears, sterilizing tools between cuts. Improve airflow by pruning overcrowded branches and 
thin canopy areas. Clear leaf litter from under trees during fall to eliminate overwintering fungal spores. These practices significantly 
reduce disease pressure in subsequent seasons.""",
        
        "pesticide": """Apply fungicides such as Captan (1.5 kg/1000L), Mancozeb (1.5 kg/1000L), or Myclobutanil every 7-10 days 
during humid conditions, starting from bud break through fruit development. For organic farming, use Sulfur 80% WP or Copper-based 
fungicides like Bordeaux mixture (1:1:100). Always follow label instructions for proper dosage, safety precautions, and withholding periods.""",
        
        "prevention": """Avoid overhead irrigation, which increases leaf wetness and fungal growth. Instead, use drip irrigation or 
water at soil level in early morning. Maintain balanced fertilization (especially potassium) to strengthen plant immunity. 
Plant disease-resistant apple varieties like Liberty, Priscilla, or William's Pride. Ensure proper orchard spacing (20-25 feet) for 
good air circulation and faster drying of leaves.""",
        
        "action": """Inspect leaves and fruits every 3-4 days during monsoon/rainy season for brown, circular lesions with concentric rings. 
Remove infected leaves immediately and dispose of them in a sealed bag or compost pit—do not leave on ground. Check nearby apple trees 
for early symptoms to prevent spread. Start preventive spraying before symptoms appear if weather conditions are humid."""
    },

    "Apple_Black_rot_Apple": {
        "remedy": """Prune out all infected branches back to healthy wood (at least 30 cm beyond visible symptoms). Disinfect pruning 
tools with 70% alcohol or 5% bleach solution between cuts to prevent disease transmission. Remove mummified fruits (dried, shriveled 
fruits) from trees and ground, as these harbor fungal spores. Cut out any cankers on the main trunk and cover wounds with wound-healing 
paste to prevent further infection.""",
        
        "pesticide": """Use fungicides like Mancozeb (1.5 kg/1000L), Carbendazim (500g/1000L), or Copper-based fungicides at 7-10 day 
intervals during fruit development. For organic control, apply Bordeaux mixture (1:1:100) or Sulfur dust. Begin applications after fruit 
set and continue until 2 weeks before harvest. Spray should thoroughly cover canopy and fruit surfaces.""",
        
        "prevention": """Maintain excellent orchard sanitation by removing dead branches and mummified fruits regularly. Ensure proper 
pruning to improve light penetration and air circulation. Avoid wounding trees during pruning or harvesting. Plant in well-drained areas 
to prevent waterlogging. Use disease-resistant rootstocks when available. Maintain balanced nitrogen fertilization (excess nitrogen increases 
disease susceptibility).""",
        
        "action": """Regularly inspect apple fruits and branches, especially during humid weather. Look for circular, sunken lesions on 
fruit with concentric zones. On branches, watch for reddish-brown cankers that may girdle the branch. Remove affected fruit immediately 
and burn or bury deeply. Prune out affected branches at first sign of disease. Monitor surrounding trees weekly."""
    },

    "Apple_Cedar_apple_rust_Apple": {
        "remedy": """Remove infected fruit and severely infected shoots. These parts produce spores that spread to nearby cedar/juniper 
trees and back to apple trees. Prune to improve air circulation and reduce leaf wetness duration. Remove any nearby juniper or cedar trees 
if possible, as they are alternate hosts essential for disease completion. Some farmers use yellow sticky traps to monitor insect vectors.""",
        
        "pesticide": """Apply fungicides like Myclobutanil (400g/1000L) or Mancozeb (1.5 kg/1000L) starting at petal fall and continuing 
at 14-21 day intervals. For organic farming, use Sulfur 80% WP at regular intervals. Spray coverage must include all leaf surfaces. 
Begin applications before symptoms appear during seasons when cedar/juniper trees are releasing spores.""",
        
        "prevention": """Identify and remove nearby cedar, juniper, or similar coniferous trees (within 3-5 km if possible) that serve as 
alternate hosts. Plant apple trees in locations with good air circulation and morning sun to dry dew quickly. Use disease-resistant varieties 
when available. Maintain balanced fertilization and proper irrigation.""",
        
        "action": """Monitor apple leaves from late spring through summer for orange or reddish spots (resembling rust). Check nearby cedar 
and juniper trees for galls (warty growths) on branches—these release spores in spring. If found, plan tree removal. Remove infected apple 
fruit and shoots promptly. Scout weekly during growing season."""
    },

    # ==== TOMATO DISEASES ====
    "Early_blight_Tomato": {
        "remedy": """Remove lower leaves on tomato plants (first 30 cm from ground) to reduce disease pressure and improve air circulation. 
Pinch off infected leaves as soon as symptoms appear. Stake or trellis plants to keep foliage off soil, where fungal spores survive. 
Mulch around plants (5 cm straw or wood chips) to prevent soil splash on leaves. Remove plant debris immediately after harvest.""",
        
        "pesticide": """Apply fungicides like Mancozeb (1.5 kg/1000L), Chlorothalonil (1.5 kg/1000L), or Azoxystrobin every 7-10 days 
starting at first sign of disease. For organic farming, use Copper-based fungicides (Bordeaux mixture 1:1:100) or Sulfur 80% WP. 
Spray foliage thoroughly, covering leaf undersides where spores germinate. Rotate with different fungicide groups to prevent resistance.""",
        
        "prevention": """Ensure 1-1.5 meter spacing between plants and adequate irrigation spacing for air circulation. Avoid overhead 
irrigation or water early in morning so leaves dry quickly. Maintain balanced nitrogen (excess promotes leaf growth and disease). 
Grow disease-resistant varieties like Iron Lady or Mountain Magic. Sanitize tools and stakes before use on new season. Mulch to reduce 
soil-to-leaf spore splash.""",
        
        "action": """Scout plants 2-3 times weekly for brown, concentric spots starting on lower leaves. Remove spotted leaves at first 
sign and dispose in sealed bag. Continue removing lower leaves as plants grow. Check underside of leaves for small spores. If disease 
appears, increase spray frequency to every 5-7 days. Maintain detailed field records of disease progression."""
    },

    "Late_blight_Tomato": {
        "remedy": """Immediately remove and destroy ALL infected leaves, stems, and fruit (burn or bury deeply; do not compost). Defoliate 
lower portion of plant to improve air circulation. Remove entire plant if more than 50% is infected. Remove crop residue immediately after 
harvest and destroy. This is a serious, fast-spreading disease—prompt action is critical.""",
        
        "pesticide": """Use preventive fungicides like Mancozeb (1.5 kg/1000L), Metalaxyl (1 kg/1000L), or fixed copper at 5-7 day intervals 
during humid weather or after rain. For organic control, use Bordeaux mixture (1:1:100) or Copper-based products. Spray BEFORE symptoms 
appear during rainy season. Tank-mix fungicides with insecticides if needed. Increase spray frequency during wet weather.""",
        
        "prevention": """Maintain wide plant spacing (1.5-2 meters) for excellent air circulation. Avoid overhead watering; use drip 
irrigation instead. Water early morning so plants dry by mid-day. Use disease-resistant varieties when available. Remove volunteer plants 
and weeds that can harbor the pathogen. Store seed potatoes in cool, dry place away from infected material. Rotate crops—avoid planting 
tomato/potato in same field for 2-3 years.""",
        
        "action": """This disease spreads RAPIDLY in wet conditions. Scout daily during rain or after rain. Look for water-soaked spots 
on leaves, stems, and green fruit. If found, increase spray frequency immediately to every 3-4 days and increase fungicide concentration 
slightly (check label). Remove entire plants if infection is severe. Burn or bury plant material—never compost. Disinfect tools and gloves 
after each plant."""
    },

    # ==== POTATO DISEASES ====
    "Potato_Early_blight_Potato": {
        "remedy": """Remove and destroy all infected leaves, stems, and volunteer potato plants from previous seasons. Prune lower foliage 
(first 30 cm) to reduce disease. Keep field free of plant debris and weeds. Increase spacing between rows for better air movement. Hill 
soil around plants to cover tubers and reduce spore splash.""",
        
        "pesticide": """Apply Mancozeb (1.5 kg/1000L), Chlorothalonil, or fixed Copper fungicides at 7-10 day intervals starting from crop 
emergence. For organic farming, use Sulfur 80% WP or Bordeaux mixture (1:1:100). Begin applications before symptoms appear if weather is 
humid. Ensure thorough coverage of entire plant including undersides of leaves. Consider alternating fungicide groups to prevent resistance.""",
        
        "prevention": """Use certified, disease-free seed potatoes from reliable sources. Maintain 45-50 cm spacing between rows. Ensure 
balanced fertilization (avoid excess nitrogen which promotes leaf growth). Plant at optimal time to avoid peak disease pressure periods. 
Improve soil drainage to reduce surface moisture. Choose disease-resistant varieties if available. Rotate crops with non-solanaceous crops 
for 2-3 years.""",
        
        "action": """Scout plants 2-3 times weekly starting from 4-6 weeks after planting. Look for concentric brown/black spots on lower 
leaves. Remove spotted leaves at first sign. Monitor weather—disease develops rapidly in warm, humid conditions. If disease appears during 
wet season, increase spray frequency to every 5 days. Keep field records of spray dates and fungicides used."""
    },

    "Potato_Late_blight_Potato": {
        "remedy": """Remove and destroy all infected plants and tubers. Destroy ALL plant debris after harvest by burning or deep burial. 
Do not leave infected material in field or compost heap. Remove volunteer potatoes immediately. Do not store any suspect tubers. This 
disease is critical for tuber storage—infected tubers rot during storage and can destroy entire crop.""",
        
        "pesticide": """Use preventive fungicides like Metalaxyl (1 kg/1000L), Mancozeb (1.5 kg/1000L), or fixed Copper at 5-7 day 
intervals during wet/humid weather. For organic farming, use Bordeaux mixture (1:1:100) or Copper products. Apply BEFORE symptoms appear. 
Spray every 3-4 days during heavy rain or immediately after rain. Check label for pre-harvest withholding period (typically 5-14 days).""",
        
        "prevention": """Use ONLY certified, disease-free seed potatoes. This is THE most important prevention measure. Ensure excellent 
field drainage to reduce soil moisture. Maintain 45-50 cm row spacing for air circulation. Avoid overhead irrigation. Water early morning 
so plants dry quickly. Remove ALL volunteer potato plants and proximal tomato crops (share same pathogen). Rotate with non-solanaceous crops 
for at least 2-3 years. Store potatoes in cool (2-4°C), well-ventilated areas.""",
        
        "action": """CRITICAL: Scout DAILY during wet season or rainy weather. Look for water-soaked spots on leaves/stems and white 
powder on leaf undersides. If found, treat immediately. Destroy infected plants completely. Increase spray frequency to 3-4 day intervals 
and maintain through harvest. After harvest, destroy all plant material—do not leave in field. Never store suspect tubers. Keep detailed 
records of all observations and treatments."""
    },

    # ==== CORN DISEASES ====
    "Corn_Common_rust_Corn": {
        "remedy": """Remove alternative hosts like grasses and sedges growing near field. Improve air circulation by maintaining proper row 
spacing and controlling weeds. In severe cases with high disease pressure, consider early harvest. Remove leaf litter from field after harvest 
to reduce overwintering spores.""",
        
        "pesticide": """Apply fungicides like Propiconazole (500 ml/1000L) or Tebuconazole (750g/1000L) at boot to grain stage when disease 
appears. For organic farming, use Sulfur 80% WP or Neem-based products. One well-timed application usually controls rust if applied early. 
Spray foliage thoroughly for good coverage.""",
        
        "prevention": """Plant rust-resistant corn hybrids available for your region. Maintain recommended row spacing for air circulation. 
Control weeds, especially grasses that serve as alternate hosts. Remove volunteer corn plants. Ensure balanced fertilization to promote plant 
vigor and disease resistance. Rotate crops with non-grass crops. Plow under crop residue after harvest to reduce overwintering spores.""",
        
        "action": """Scout field weekly starting from boot stage for small, reddish-brown pustules on leaf surfaces. Once found, begin 
monitoring more frequently. If disease is on more than 50% of leaf area before grain fill, consider fungicide application. Check resistant 
varieties and plant those in future seasons. Maintain field records of disease observations and hybrid performance."""
    },

    "Corn_Northern_leaf_blight_Corn": {
        "remedy": """Remove and destroy corn residue after harvest by plowing or burning to eliminate fungal spores. Remove volunteer corn 
plants that harbor the pathogen. Thin corn stands if overcrowded to improve air circulation and light penetration. Remove lower leaves if 
severely affected to reduce disease spread.""",
        
        "pesticide": """Apply fungicides like Azoxystrobin (200ml/1000L), Propiconazole (500ml/1000L), or Pyraclostrobin at V10-V12 stage 
(10-12 leaves visible) and repeat if needed at boot stage. For organic farming, use Copper or Sulfur-based products. Application timing is 
critical—apply BEFORE extensive leaf damage occurs. One or two well-timed applications usually suffice.""",
        
        "prevention": """This is the PRIMARY prevention strategy: Plant only hybrid corn that is resistant to Northern Leaf Blight 
(resistant hybrids are widely available). Rotate crops—avoid continuous corn production. Ensure adequate spacing (60-75 cm rows) for air 
circulation. Maintain balanced nitrogen fertilization. Control weeds that can shade plants and increase humidity. Plow residue immediately 
after harvest.""",
        
        "action": """Scout field starting at V6-V8 stage and continue weekly. Look for long, rectangular, gray-brown lesions on leaves, often 
starting on lower leaves. Monitor weather—disease develops rapidly in warm (24-27°C), wet conditions. If disease appears during grain fill, 
consider fungicide application. Select resistant hybrids for next season based on local disease pressure records."""
    },

    # ==== PEPPER/CHILLI DISEASES ====
    "Pepper_bell_Bacterial_spot_Pepper_bell": {
        "remedy": """Remove and destroy all infected plant parts—diseased leaves, stems, and fruit. For severely infected plants, consider 
complete removal. Prune to improve air circulation. Disinfect all tools and equipment with 70% alcohol between plants. Remove any nearby 
weed hosts.""",
        
        "pesticide": """For bacterial diseases, use copper-based bactericides like Copper Sulfate (1 kg/1000L) or Bordeaux mixture (1:1:100) 
at 10-14 day intervals. Spray thoroughly, covering all leaf surfaces and fruit. There is NO fungicide effective for bacterial diseases—use 
only copper products. Spray preventively starting before symptoms appear during humid season.""",
        
        "prevention": """Use disease-free seeds and nursery plants from certified sources. Maintain wide plant spacing (60-75 cm) for 
excellent air circulation. Avoid overhead watering; use drip irrigation instead. Water early morning so plants dry quickly. Remove weeds 
that can harbor bacteria. Maintain field sanitation—remove plant debris immediately. Rotate crops with non-solanaceous crops for 2-3 years. 
Do NOT save seed from infected plants.""",
        
        "action": """Scout plants 2-3 times weekly for small, dark, water-soaked spots on leaves and fruit. Bacterial spot spreads rapidly 
in wet weather. Remove infected leaves at first sign and dispose in sealed bag. If disease spreads rapidly, increase spray frequency. Disinfect 
hands and tools after handling infected plants. Keep detailed records of disease progression."""
    },

    # ==== GENERAL HEALTHY PLANT ====
    "healthy_Plant": {
        "remedy": """Your plant is healthy! No disease treatment is required at this time. Continue with regular plant care and maintenance 
practices to keep it disease-free. Monitor regularly for any signs of stress or disease development.""",
        
        "pesticide": """No pesticide application is needed for a healthy plant. Focus on prevention through good cultural practices. Unnecessary 
pesticide applications increase costs, harm beneficial insects, and can leave residues on produce.""",
        
        "prevention": """Maintain these excellent practices: (1) Ensure proper spacing between plants for air circulation, (2) Use drip 
irrigation or water at soil level in early morning, (3) Maintain balanced fertilization appropriate for your crop, (4) Scout regularly for 
early disease signs, (5) Remove weeds and plant debris promptly, (6) Rotate crops annually, (7) Use disease-resistant varieties when available.""",
        
        "action": """Maintenance tips: Water deeply but infrequently to encourage deep root development. Apply 5 cm organic mulch around 
plants to conserve moisture and suppress weeds. Fertilize according to soil test recommendations or crop-specific schedules. Prune to improve 
light penetration and air circulation. Scout plant health weekly to catch any disease development early. Keep field records of all activities 
and observations."""
    },

    # ==== ADDITIONAL COMMON DISEASES ====
    "Powdery_mildew_Apple": {
        "remedy": """Remove infected leaves and shoots. Prune to open up plant canopy and improve air circulation. In severe cases, consider 
defoliation of heavily infected branches. Avoid working in plant when wet to prevent spore spread. Regular removal of infected plant parts 
early in season is very effective.""",
        
        "pesticide": """Apply Sulfur 80% WP (2 kg/1000L) or fungicides like Triazoles (Myclobutanil 400g/1000L, Hexaconazole 500g/1000L) at 
7-10 day intervals during dry season. For organic farming, Sulfur is the preferred choice. Spray until all leaf surfaces are covered but not 
dripping. Avoid spraying during hot weather (above 32°C) with sulfur products.""",
        
        "prevention": """Maintain excellent air circulation through proper spacing and pruning. Avoid excess nitrogen fertilization, which 
promotes tender, susceptible growth. Ensure good drainage around plants. In nurseries, use preventive spray schedules. Select powdery mildew 
resistant varieties if available.""",
        
        "action": """Scout regularly for white, powdery coating on leaves, stems, and fruit. This coating contains spores. Remove infected 
leaves at first sign. If disease develops, begin spraying immediately. Monitor closely as spores spread rapidly in cool-warm seasons (15-24°C)."""
    },

    "Septoria_leaf_spot_Tomato": {
        "remedy": """Remove lower leaves (first 30-45 cm from ground) to prevent splash from soil. Prune infected leaves immediately upon 
detection. Stake or trellis plants to keep foliage above soil. Remove all plant debris after harvest and destroy it—do not compost.""",
        
        "pesticide": """Apply Chlorothalonil (1.5 kg/1000L), Mancozeb (1.5 kg/1000L), or Carbendazim (500g/1000L) at 10-14 day intervals. 
For organic farming, use Copper-based fungicides or Sulfur products. Begin preventive sprays before symptoms appear during humid season. 
Rotate fungicide classes to prevent resistance development.""",
        
        "prevention": """Ensure 1.5-2 meter spacing for air circulation. Use drip irrigation, not overhead spray. Water early morning so 
plants dry quickly. Maintain mulch (5 cm) around plants to prevent soil splash. Use disease-resistant varieties where available. Sanitize 
stakes and tools before use. Remove volunteer plants and weeds. Practice strict crop rotation—avoid planting tomato in same field for 2-3 
years.""",
        
        "action": """Scout regularly for small, circular spots with concentric rings starting on lower leaves. Spots have pycnidia (tiny, 
flask-shaped structures) visible in center. Remove spotted leaves at first sign. If disease appears, increase spray frequency. Keep field 
records to track disease progression and fungicide effectiveness."""
    },
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_disease_info(disease_key: str) -> dict:
    """
    Retrieve detailed information for a specific disease.
    
    Args:
        disease_key: Disease key in format "Disease_Crop" or "Disease_Crop_additional"
        
    Returns:
        Dictionary with remedy, pesticide, prevention, action
        Returns generic info if disease not found
    """
    # Try exact match first
    if disease_key in DISEASE_DATABASE:
        return DISEASE_DATABASE[disease_key]
    
    # Try with different variations
    for key in DISEASE_DATABASE.keys():
        if disease_key.lower().replace(' ', '_') in key.lower():
            return DISEASE_DATABASE[key]
    
    # Return generic information if not found
    return {
        "remedy": """Disease information not yet available in our database. 
Please consult with your local agricultural extension office or plant pathologist for specific guidance. 
In the meantime, practice good field sanitation and remove infected plant parts.""",
        
        "pesticide": """Please consult with your local agricultural officer for region-specific fungicide recommendations. 
Product availability and effectiveness varies by region and season. Always follow product label instructions.""",
        
        "prevention": """General prevention practices: Ensure proper plant spacing for air circulation, use disease-free seeds/planting material, 
practice crop rotation, maintain field sanitation, use resistant varieties when available, and monitor plants regularly for disease signs.""",
        
        "action": """Contact your local agricultural extension office or plant pathologist for accurate diagnosis and treatment recommendations. 
Proper disease identification is critical for effective management."""
    }


def get_healthy_plant_info() -> dict:
    """
    Get information for a healthy plant.
    
    Returns:
        Dictionary with maintenance and care information
    """
    return DISEASE_DATABASE.get("healthy_Plant", {
        "remedy": "Your plant is healthy! No treatment needed.",
        "pesticide": "No pesticide needed for healthy plants.",
        "prevention": "Continue good cultural practices.",
        "action": "Monitor regularly and maintain current care practices."
    })


# ============================================================================
# DISEASE CATEGORIES (for UI grouping)
# ============================================================================

DISEASE_CATEGORIES = {
    "Apple": ["Apple_scab_Apple", "Apple_Black_rot_Apple", "Apple_Cedar_apple_rust_Apple", "Powdery_mildew_Apple"],
    "Tomato": ["Early_blight_Tomato", "Late_blight_Tomato", "Septoria_leaf_spot_Tomato"],
    "Potato": ["Potato_Early_blight_Potato", "Potato_Late_blight_Potato"],
    "Corn": ["Corn_Common_rust_Corn", "Corn_Northern_leaf_blight_Corn"],
    "Pepper": ["Pepper_bell_Bacterial_spot_Pepper_bell"]
}

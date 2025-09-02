def segment_data_with_indices(data, threshold):
    """
    Segments the data based on the difference between consecutive values and returns the indices of each segment.
    
    :param data: List of numerical data points
    :param threshold: The difference threshold for segmentation
    :return: A list of tuples, each containing the segment values and their corresponding indices
    """
    if not data:
        return []

    segments = []
    current_segment = [data[0]]
    current_indices = [0]

    for i in range(1, len(data)):
        # Check if the difference between consecutive values is less than the threshold
        if data[i] - data[i - 1] < threshold:
            segments.append((current_segment, current_indices))
            current_segment = [data[i]]
            current_indices = [i]
        else:
            current_segment.append(data[i])
            current_indices.append(i)
            

    # Append the last segment
    if current_segment:
        segments.append((current_segment, current_indices))

    return segments

# Example usage
data = [ #32 layer wikitext2 dataset
    71.09449688487334, 70.85130519012856, 67.40613406287716, 69.04946102517368,
    69.22327756162719, 69.51528308727653, 71.96004949780146, 73.14870527360563,
    72.39993716987092, 73.33161718592027, 73.3456552653242, 73.32481194641586,
    73.68638390929002, 73.93574881680618, 74.61757474828251, 73.30678839776678,
    73.6027615610652, 73.91468930567065, 73.90614702585596, 74.62802834396467,
    74.46465621554732, 74.76733693669686, 74.94375721541391, 75.72328136974946,
    74.83840156524144, 76.99661199571894, 75.59826604347896, 76.24339032859046,
    76.6184171897687, 75.7592322159148, 75.27747141727312, 73.44524688370723
]

threshold = -1

data = [ #40 layer wikitext2 dataset
    88.56142582324127, 83.1908786508794, 84.38408339204648, 83.21636804954474, 
    86.92246700703916, 87.53768984363012, 89.18191230248249, 89.19035446942016, 
    89.9533398679659, 90.96597146195018, 91.86337784763947, 92.51877033291755, 
    92.03969198399568, 92.46515476042592, 93.2050576149498, 91.42414335910942, 
    92.64186295234181, 93.00824597973778, 92.98066834434528, 93.16376468223461, 
    93.37936712233315, 93.11623838602648, 94.16686739888758, 93.08618498622621, 
    92.58229511268657, 92.879676711689, 93.11112275027877, 94.19428981060773, 
    94.16337504479995, 94.64360385186022, 96.26818667060098, 94.77489512296752, 
    95.58909909940053, 95.22007943142523, 95.31731525843452, 94.19310590088801, 
    95.28644231266685, 92.50816799468708, 92.20046121083985, 92.14139987322815
]

# Define the threshold for segmentation
threshold = -1

# Segment the data
segmented_data = segment_data_with_indices(data, threshold)

# Print the segmented data and their indices
for segment, indices in segmented_data:
    print(f"Indices: {indices}")

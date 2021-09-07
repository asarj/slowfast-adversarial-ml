import argparse
import os
from pathlib import Path
import pandas as pd


VIDEO_CLASSES = {"sharpening knives": 290, "eating ice cream": 115, "cutting nails": 81, "changing wheel": 53, "bench pressing": 19, "deadlifting": 88, "eating carrots": 111, "marching": 192, "throwing discus": 358, "playing flute": 231, "cooking on campfire": 72, "breading or breadcrumbing": 33, "playing badminton": 218, "ripping paper": 276, "playing saxophone": 244, "milking cow": 197, "juggling balls": 169, "flying kite": 130, "capoeira": 43, "making jewelry": 187, "drinking": 100, "playing cymbals": 228, "cleaning gutters": 61, "hurling (sport)": 161, "playing organ": 239, "tossing coin": 361, "wrestling": 395, "driving car": 103, "headbutting": 150, "gymnastics tumbling": 147, "making bed": 186, "abseiling": 0, "holding snake": 155, "rock climbing": 278, "cooking egg": 71, "long jump": 182, "bee keeping": 17, "trimming or shaving beard": 365, "cleaning shoes": 63, "dancing gangnam style": 86, "catching or throwing softball": 50, "ice skating": 164, "jogging": 168, "eating spaghetti": 116, "bobsledding": 28, "assembling computer": 8, "playing cricket": 227, "playing monopoly": 238, "golf putting": 143, "making pizza": 188, "javelin throw": 166, "peeling potatoes": 211, "clapping": 57, "brushing hair": 36, "flipping pancake": 129, "drinking beer": 101, "dribbling basketball": 99, "playing bagpipes": 219, "somersaulting": 325, "canoeing or kayaking": 42, "riding unicycle": 275, "texting": 355, "tasting beer": 352, "hockey stop": 154, "playing clarinet": 225, "waxing legs": 389, "curling hair": 80, "running on treadmill": 281, "tai chi": 346, "driving tractor": 104, "shaving legs": 293, "sharpening pencil": 291, "making sushi": 190, "spray painting": 327, "situp": 305, "playing kickball": 237, "sticking tongue out": 331, "headbanging": 149, "folding napkins": 132, "playing piano": 241, "skydiving": 312, "dancing charleston": 85, "ice fishing": 163, "tickling": 359, "bandaging": 13, "high jump": 151, "making a sandwich": 185, "riding mountain bike": 271, "cutting pineapple": 82, "feeding goats": 125, "dancing macarena": 87, "playing basketball": 220, "krumping": 179, "high kick": 152, "balloon blowing": 12, "playing accordion": 217, "playing chess": 224, "hula hooping": 159, "pushing wheelchair": 263, "riding camel": 268, "blowing out candles": 27, "extinguishing fire": 121, "using computer": 373, "jumpstyle dancing": 173, "yawning": 397, "writing": 396, "jumping into pool": 172, "doing laundry": 96, "egg hunting": 118, "sanding floor": 284, "moving furniture": 200, "exercising arm": 119, "sword fighting": 345, "sign language interpreting": 303, "counting money": 74, "bartending": 15, "cleaning windows": 65, "blasting sand": 23, "petting cat": 213, "sniffing": 320, "bowling": 31, "playing poker": 242, "taking a shower": 347, "washing hands": 382, "water sliding": 384, "presenting weather forecast": 254, "tobogganing": 360, "celebrating": 51, "getting a haircut": 138, "snorkeling": 321, "weaving basket": 390, "playing squash or racquetball": 245, "parasailing": 206, "news anchoring": 202, "belly dancing": 18, "windsurfing": 393, "braiding hair": 32, "crossing river": 78, "laying bricks": 181, "roller skating": 280, "hopscotch": 156, "playing trumpet": 248, "dying hair": 108, "trimming trees": 366, "pumping fist": 256, "playing keyboard": 236, "snowboarding": 322, "garbage collecting": 136, "playing controller": 226, "dodgeball": 94, "recording music": 266, "country line dancing": 75, "dancing ballet": 84, "gargling": 137, "ironing": 165, "push up": 260, "frying vegetables": 135, "ski jumping": 307, "mowing lawn": 201, "getting a tattoo": 139, "rock scissors paper": 279, "cheerleading": 55, "using remote controller (not gaming)": 374, "shaking head": 289, "sailing": 282, "training dog": 363, "hurdling": 160, "fixing hair": 128, "climbing ladder": 67, "filling eyebrows": 126, "springboard diving": 329, "eating watermelon": 117, "drumming fingers": 106, "waxing back": 386, "playing didgeridoo": 229, "swimming backstroke": 339, "biking through snow": 22, "washing feet": 380, "mopping floor": 198, "throwing ball": 357, "eating doughnuts": 113, "drinking shots": 102, "tying bow tie": 368, "dining": 91, "surfing water": 337, "sweeping floor": 338, "grooming dog": 145, "catching fish": 47, "pumping gas": 257, "riding or walking with horse": 273, "massaging person's head": 196, "archery": 5, "ice climbing": 162, "playing recorder": 243, "decorating the christmas tree": 89, "peeling apples": 210, "snowmobiling": 324, "playing ukulele": 249, "eating burger": 109, "building cabinet": 38, "stomping grapes": 332, "drop kicking": 105, "passing American football (not in game)": 209, "applauding": 3, "hugging": 158, "eating hotdog": 114, "pole vault": 253, "reading newspaper": 265, "snatch weight lifting": 318, "zumba": 399, "playing ice hockey": 235, "breakdancing": 34, "feeding fish": 124, "shredding paper": 300, "catching or throwing frisbee": 49, "exercising with an exercise ball": 120, "pushing cart": 262, "swimming butterfly stroke": 341, "riding scooter": 274, "spraying": 328, "folding paper": 133, "golf driving": 142, "robot dancing": 277, "bending back": 20, "testifying": 354, "waxing chest": 387, "carving pumpkin": 46, "hitting baseball": 153, "riding elephant": 269, "brushing teeth": 37, "pull ups": 255, "riding a bike": 267, "skateboarding": 306, "cleaning pool": 62, "playing paintball": 240, "massaging back": 193, "shoveling snow": 299, "surfing crowd": 336, "unboxing": 371, "faceplanting": 122, "trapezing": 364, "swinging legs": 343, "hoverboarding": 157, "playing violin": 250, "wrapping present": 394, "blowing nose": 26, "kicking field goal": 174, "picking fruit": 214, "swinging on something": 344, "giving or receiving award": 140, "planting trees": 215, "water skiing": 383, "washing dishes": 379, "punching bag": 258, "massaging legs": 195, "throwing axe": 356, "salsa dancing": 283, "bookbinding": 29, "tying tie": 370, "skiing crosscountry": 309, "shining shoes": 295, "making snowman": 189, "front raises": 134, "doing nails": 97, "massaging feet": 194, "playing drums": 230, "smoking": 316, "punching person (boxing)": 259, "cartwheeling": 45, "passing American football (in game)": 208, "shaking hands": 288, "plastering": 216, "watering plants": 385, "kissing": 176, "slapping": 314, "playing harmonica": 233, "welding": 391, "smoking hookah": 317, "scrambling eggs": 285, "cooking chicken": 70, "pushing car": 261, "opening bottle": 203, "cooking sausages": 73, "catching or throwing baseball": 48, "swimming breast stroke": 340, "digging": 90, "playing xylophone": 252, "doing aerobics": 95, "playing trombone": 247, "knitting": 178, "waiting in line": 377, "tossing salad": 362, "squat": 330, "vault": 376, "using segway": 375, "crawling baby": 77, "reading book": 264, "motorcycling": 199, "barbequing": 14, "cleaning floor": 60, "playing cello": 223, "drawing": 98, "auctioning": 9, "carrying baby": 44, "diving cliff": 93, "busking": 41, "cutting watermelon": 83, "scuba diving": 286, "riding mechanical bull": 270, "making tea": 191, "playing tennis": 246, "crying": 79, "dunking basketball": 107, "cracking neck": 76, "arranging flowers": 7, "building shed": 39, "golf chipping": 141, "tasting food": 353, "shaving head": 292, "answering questions": 2, "climbing tree": 68, "skipping rope": 311, "kitesurfing": 177, "juggling fire": 170, "laughing": 180, "paragliding": 205, "contact juggling": 69, "slacklining": 313, "arm wrestling": 6, "making a cake": 184, "finger snapping": 127, "grooming horse": 146, "opening present": 204, "tapping pen": 351, "singing": 304, "shot put": 298, "cleaning toilet": 64, "spinning poi": 326, "setting table": 287, "tying knot (not on a tie)": 369, "blowing glass": 24, "eating chips": 112, "tap dancing": 349, "climbing a rope": 66, "brush painting": 35, "chopping wood": 56, "stretching leg": 334, "petting animal (not cat)": 212, "baking cookies": 11, "stretching arm": 333, "beatboxing": 16, "jetskiing": 167, "bending metal": 21, "sneezing": 319, "folding clothes": 131, "sled dog racing": 315, "tapping guitar": 350, "bouncing on trampoline": 30, "waxing eyebrows": 388, "air drumming": 1, "kicking soccer ball": 175, "washing hair": 381, "riding mule": 272, "blowing leaves": 25, "strumming guitar": 335, "playing cards": 222, "snowkiting": 323, "playing bass guitar": 221, "applying cream": 4, "shooting basketball": 296, "walking the dog": 378, "triple jump": 367, "shearing sheep": 294, "clay pottery making": 58, "bungee jumping": 40, "unloading truck": 372, "shuffling cards": 301, "shooting goal (soccer)": 297, "tango dancing": 348, "side kick": 302, "grinding meat": 144, "yoga": 398, "hammer throw": 148, "changing oil": 52, "checking tires": 54, "parkour": 207, "eating cake": 110, "skiing slalom": 310, "juggling soccer ball": 171, "whistling": 392, "feeding birds": 123, "playing volleyball": 251, "swing dancing": 342, "skiing (not slalom or crosscountry)": 308, "lunge": 183, "disc golfing": 92, "clean and jerk": 59, "playing guitar": 232, "baby waking up": 10, "playing harp": 234}


def main(
    split, path_to_videos, path_to_videos_csv, output_filename
):
    # Get full Kinetics 400 metadata
    print(f"Extracting full Kinetics 400 CSV from {path_to_videos_csv}...")
    videos_df = pd.read_csv(
        path_to_videos_csv
    )
    
    # Get all downloaded videos
    print(f"Getting all downloaded videos from {path_to_videos}...")
    videos_in_dir = list(Path(path_to_videos).rglob("*.mp4"))
    videos_in_dir = [os.path.basename(f) for f in videos_in_dir]
    
    # Create new dataframe to store the csv info
    final_df = pd.DataFrame(columns=["path", "class"])
    
    print("Begin data cleaning...")
    for video in videos_in_dir:
        # Extract YouTube ID, timestamps, file extension
        v = video.split("_")
        last = v.pop()
        v.extend(last.split("."))
        timestamps = "_".join(v[-3:-1])
        ext = v[-1]
        youtube_id = "_".join(v[0:-3])
    #     print(youtube_id, timestamps, ext)
        print(f"Getting information for YouTube video ID: {youtube_id}")
        
        # Get the corresponding label for this ID
        label = list((videos_df.loc[
            (videos_df["youtube_id"] == youtube_id)    
        ]["label"]))[0]
        
        # Get the numerical form of the class
        label_id = VIDEO_CLASSES[label]
        
        # Add to dataframe
        final_df = final_df.append({
            "path": f"./kinetics-400-dataset-files/{split}/{youtube_id}_{timestamps}.{ext}",
            "class": label_id
        }, ignore_index=True)
        
    # Save everything
    print(f"Saving the data to {output_filename}")
    final_df.to_csv(output_filename, header=False, index=False, sep=' ')


if __name__ == '__main__':
    description = 'Helper script for generating the csv that SlowFast uses to train and test'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('-split', "--split", type=str,
                   help=('Type of partition of the dataset, one of {train, test, val}'))
    p.add_argument('-path_to_videos', "--path-to-videos", type=str,
                   help='Path to root directory containing all downloaded videos from Kinetics400')
    p.add_argument('-path_to_videos_csv', "--path-to-videos-csv", type=str,
                   help='Path to csv file containing all Kinetics 400 video metadata')
    p.add_argument('-output_filename', "-o", type=str,
                   help='Path to save the cleaned csv to, which contains paths to the downloaded videos and their numerical classes')
    main(**vars(p.parse_args()))
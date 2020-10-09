import glob
import os
import shutil

def process_file(path, output_dir):
    """
    Process a single frequencies file and save it into
    a single line file with the next format:
    n\tl\tm\tvalue ....
    """
    line_out = []
    dirname_output = path.split("/")[-2]
    filename_ouput = path.split("/")[-1]
    sep = "\t"
    with open(path, "r") as infile:
        for id, row in enumerate(infile):
            if id > 24:
                chunks = " ".join(row.split()).replace(" ", ",").split(",")
                if len(chunks) == 8:  # Info Star at 8 line
                    line_out.append(chunks[0])
                    line_out.append(sep)
                    line_out.append(chunks[1])
                    line_out.append(sep)
                    line_out.append(chunks[2])
                    line_out.append(sep)
                    line_out.append(chunks[3])
                    line_out.append(sep)
    print("Procesed % freqs" % len(line_out))
    # Check if directory exists, and overwrite if it is
    if not os.path.exists(output_dir + dirname_output):
        #shutil.rmtree(output_dir + dirname_output)
        os.makedirs(output_dir + dirname_output)
    # Write to file
    with open(output_dir + dirname_output + "/" +filename_ouput, "w") as outfile:
        outfile.write("".join(line_out))

# Output dir to save all models
output_dir = "/home/roberto/Downloads/evolutionTracks_line/"
filou_folder = "/home/roberto/Downloads/evolutionTracks/FILOU/*

filou_dirs = glob.glob(filou_folder)
for filou_dir in filou_dirs:
    print("Processing" + filou_dir)
    for file in glob.glob(filou_dir + "/*.frq"):
        print(" " + file)
        process_file(file, output_dir)

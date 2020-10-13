import glob
import os
import shutil


def process_file(path, output_dir, MAX_FREQS_PROCESSED=503):
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
                    n = int(chunks[0])
                    l = int(chunks[1])
                    m = int(chunks[2])
                    freq = float(chunks[3])
                    freq *= 0.0864  # to muHZ
                    if n > 0 and n < 10 and l < 3:
                        line_out.append(n)
                        line_out.append(sep)
                        line_out.append(l)
                        line_out.append(sep)
                        line_out.append(m)
                        line_out.append(sep)
                        line_out.append(freq)
                        line_out.append(sep)
    # Check if directory exists, and overwrite if it is
    if not os.path.exists(output_dir + dirname_output):
        # shutil.rmtree(output_dir + dirname_output)
        os.makedirs(output_dir + dirname_output)
    if len(line_out) < MAX_FREQS_PROCESSED:
        print(
            "Not enough frequencies (%d) for star %s"
            % (len(line_out), dirname_output + "-" + filename_ouput)
        )
    else:
        print(
            "Procesed %d freqs (keeping only %d)" % (len(line_out), MAX_FREQS_PROCESSED)
        )
        # Write to file
        with open(output_dir + dirname_output + "/" + filename_ouput, "w") as outfile:
            outfile.write(
                "".join(str(v) for v in line_out[0:MAX_FREQS_PROCESSED])
            )  # Map al list to str before join and save


# Output dir to save all models
output_dir = "/home/roberto/Downloads/evolutionTracks_line/"
filou_folder = "/home/roberto/Downloads/evolutionTracks/FILOU/*"

MAX_FREQS_PROCESSED = 503

filou_dirs = glob.glob(filou_folder)
for filou_dir in filou_dirs:
    print("Processing" + filou_dir)
    for file in glob.glob(filou_dir + "/*.frq"):
        process_file(file, output_dir, MAX_FREQS_PROCESSED)

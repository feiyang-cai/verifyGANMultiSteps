import re

input_file_path = "./single_cell_reachable_set_converged.tex"
output_file_path = "./single_cell_reachable_set_round_converged.tex"

with open(input_file_path, "r") as input_file, open(output_file_path, "w") as output_file:
    for line in input_file:
        # Use regular expression to find all floating-point numbers in the line
        # and replace them with the rounded versions
        #match = re.match(r"(\d+\.\d+) (\d+\.\d+)", line)
        match = re.match(r"(-?\d+\.\d+) (-?\d+\.\d+)", line)
        if match:
            modified_line = f"{float(match.group(1)):.2f} {float(match.group(2)):.2f}\n"
        else:
            modified_line = line
        
        #modified_line = re.sub(r"\d+\.\d+", round_floats_to_2_decimal_places, line)
        
        # Write the modified line to the output file
        output_file.write(modified_line)
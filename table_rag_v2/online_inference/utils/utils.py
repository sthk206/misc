import json
import csv

def read_in(file_path) :
    """
    Read in json files.
    """
    with open(file_path, "r", encoding="utf-8") as fin :
        data = json.load(fin)
    return data


def read_in_lines(file_path) :
    """
    Read in dicts in lines.
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as fin :
        for idx, line in enumerate(fin) :
            try :
                data.append(json.loads(line))
            except :
                continue
    return data


def read_csv(file_path) :
    """
    Read in csv files.
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as file :
        lines = file.readlines()
    
    if not lines :
        return []
    
    header_line = lines[0].strip()
    headers = header_line.split("\t")

    for i in range(1, len(lines)) :
        line = line[i].strip()
        if line :
            values = line.split("\t")
            row_dict = {}
            for j in range(min(len(headers), len(values))) :
                row_dict[headers[j]] = values[j]
            data.append(row_dict)
    return data

def read_jsonl_file(file_path) :
    with open(file_path, "r", encoding='utf-8') as fin :
        lines = fin.readlines()
    return lines

def read_plain_csv(file_path) :
    """
    Read a csv fiel and convert it into a markdown table format.

    Args:
        file_path (str): Path to the csv file

    Returns:
        str: Markdown formatted table
    """

    try :
        with open(file_path, "r", encoding="utf-8") as f :
            reader = csv.reader(f)
            rows = list(reader)

            if not rows :
                return "Empty CSV file"
            
            headers = rows[0]
            data_rows = rows[1:]

            # Add header row
            markdown_table = []
            markdown_table.append("| " + " | ".join(headers) + " |")

            # Add separator row
            markdown_table.append("| " + " | ".join(["---" for _ in headers]) + " |")

            # Add data row
            for row in data_rows :
                markdown_table.append("| " + " | ".join(row) + " |")

            return "\n".join(markdown_table)

    except FileNotFoundError :
        return f"Error: File {file_path} not Found"
    except Exception as e :
        return f"Error reading CSV file: {str(e)}"
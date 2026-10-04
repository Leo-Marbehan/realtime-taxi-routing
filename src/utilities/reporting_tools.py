from src.utilities import result_columns as col
from src.utilities.enums import Algorithm, SolutionMode


def print_dict_as_table(input_dict):
    """Function: print a dictionary in a tabular format
    """
    # Find the maximum length of the keys for formatting
    max_key_length = max(len(str(key)) for key in input_dict.keys())
    max_value_length = max(len(str(value)) for value in input_dict.values())

    header_key = "Attribute"
    header_value = "Value"
    max_key_length = max(max_key_length, len(header_key))
    max_value_length = max(max_value_length, len(header_value))

    # Print table headers
    print(f"{header_key}{' ' * (max_key_length - len(header_key) + 2)} | {header_value}")
    print("-" * (max_key_length + 2) + "+" + "-" * (max_value_length + 2))

    # Print each item in the dictionary
    for key, value in input_dict.items():
        # Adjust spacing based on key and value length
        key_spacing = " " * (max_key_length - len(str(key)) + 2)
        print(f"{key}{key_spacing} | {value}")

def print_result_as_table(results: dict):
    """
    Print a selected subset of a results dictionary in a neat 2-column table.
    """
    legacy_fallbacks = {
        col.NUM_TRIPS: col.NUM_TRIPS_LEGACY,
        col.NUM_VEHICLES: col.NUM_VEHICLES_LEGACY,
    }
    rows = [
        (name, results.get(name, results.get(legacy_fallbacks.get(name), "")))
        for name in col.DISPLAY_COLUMNS
    ]

    # Convert values to strings (safe for None, numbers, etc.)
    rows = [(k, "" if v is None else str(v)) for k, v in rows]

    # Column widths
    left_w = max(len(k) for k, _ in rows)
    right_w = max(len(v) for _, v in rows)

    # Print table
    print(f"{'Attribute'.ljust(left_w)} | {'Value'.ljust(right_w)}")
    print(f"{'-' * left_w}-+-{'-' * right_w}")
    for k, v in rows:
        print(f"{k.ljust(left_w)} | {v}")


def create_solution_description(row):
    if row[col.SOLUTION_MODE] == SolutionMode.PARTIAL_ONLINE.value:
        if row[col.KNOWN_PORTION_PCT] == 0:
            return "Fully Online"
        else:
            return f"Partial Online ({row[col.KNOWN_PORTION_PCT]}%)"
#    elif row[col.SOLUTION_MODE] == 'advance_notice':
#        return f"Advance Notice ({row['Advance Notice (min)']} min)"
    else:
        return row[col.SOLUTION_MODE]

def merge_algorithms_param(row):
    if row[col.ALGORITHM] == Algorithm.CONSENSUS.value:
 #       return f"{row[col.CONSENSUS_TYPE]}"
        return f"Consensus ({row[col.CONSENSUS_TYPE]})"
    elif row[col.ALGORITHM] == Algorithm.RE_OPTIMIZE.value:
        return f"LNS ({row[col.DESTROY_METHOD]})"
    else:
        return row[col.ALGORITHM]


def add_data_labels(ax, metric, y_min, y_max, threshold=0.15):
    """
    Adds data labels to the bars in a bar plot with intelligent positioning.

    Parameters:
        ax (matplotlib.axes.Axes): The Axes object to add labels to.
        metric (str): The name of the metric being plotted, used for formatting.
        y_min (float): The minimum y-axis limit.
        y_max (float): The maximum y-axis limit.
        threshold (float, optional): The ratio of the y-axis range to determine label placement.
                                     Defaults to 0.15 (15%).

    Returns:
        None
    """
    for container in ax.containers:
        # Iterate over each bar in the container
        for bar in container:
            height = bar.get_height()

            # Define the threshold based on the y-axis range
            if y_max - y_min == 0:
                # Prevent division by zero
                threshold_ratio = 0
            else:
                threshold_ratio = (height - y_min) / (y_max - y_min)

            # Determine label position based on threshold
            if threshold_ratio > 2 * threshold:
                # Place label inside the bar
                label_y = height - (0.05 * (y_max - y_min))  # Slightly below the top
                va = 'top'
                color = 'white'
            else:
                # Place label outside the bar
                label_y = height + (0.02 * (y_max - y_min))  # Slightly above the bar
                va = 'bottom'
                color = 'black'

            # Format the label based on the metric
            if y_max > 100:
                label = f"{height:.0f}"
            else:
                label = f"{height:.2f}"

            # Add the label to the bar
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                label_y,
                label,
                ha='center',
                va=va,
                fontsize=7,
                color='darkslategray',
                rotation=90,
                fontweight='bold',
            )

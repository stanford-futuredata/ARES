import pandas as pd
import os

def filter_tsv_by_label(tsv_file, label, output_file=None):
    print(f"Reading the TSV file: {tsv_file}")
    # Read the TSV file into a DataFrame
    df = pd.read_csv(tsv_file, sep='\t')
    
    print(f"Filtering out rows where the label '{label}' has empty values")
    # Filter out rows where the specified label has empty values
    initial_row_count = len(df)
    filtered_df = df[df[label].notna()]
    final_row_count = len(filtered_df)
    rows_removed = initial_row_count - final_row_count
    print(f"Number of rows removed: {rows_removed}")
    print(f"Number of rows left: {final_row_count}")
    
    print(f"Removing the label column: {label}")
    # Remove the label column
    filtered_df = filtered_df.drop(columns=[label])
    
    # Remove other label columns
    other_labels = ["Context_Relevance_Label", "Answer_Relevance_Label", "Answer_Faithfulness_Label"]
    other_labels.remove(label)
    removed_labels = [col for col in other_labels if col in filtered_df.columns]
    print(f"Other label columns being removed: {removed_labels}")
    filtered_df = filtered_df.drop(columns=removed_labels)
    
    # Determine the output file path
    if output_file is None:
        base, ext = os.path.splitext(os.path.basename(tsv_file))
        output_file = f"{base}_filtered_{label}{ext}"
    
    print(f"Saving the filtered DataFrame to: {output_file}")
    # Save the filtered DataFrame to a new TSV file in the current directory
    filtered_df.to_csv(output_file, sep='\t', index=False)
    print("Filtering and saving process completed.")

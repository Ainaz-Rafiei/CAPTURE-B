import pandas as pd

def load_metadata(metadata_path, clinical_path):
    tcia_metadata = pd.read_csv(metadata_path)
    clinical_df = pd.read_csv(clinical_path, sep='\t', low_memory=False)

    tcia_metadata['case_id'] = tcia_metadata['Subject ID'].str.upper()
    clinical_df['case_id'] = clinical_df['cases.submitter_id'].str.upper()

    return tcia_metadata, clinical_df


def merge_data(metadata_df, clinical_df):
    merged_df = pd.merge(metadata_df, clinical_df, on='case_id', how='right')
    print(f"Merged rows: {len(merged_df)}")
    return merged_df


def filter_and_label_data(clinical_df):
    possible_cols = [
        'diagnoses.diagnosis',
        'diagnoses.primary_diagnosis',
        'diagnoses.histological_type',
        'cases.submitter_id'
    ]
    
    for col in possible_cols:
        if col in clinical_df.columns:
            diagnosis_col = col
            break
    else:
        raise ValueError("No known histological diagnosis column found!")
    
    # Filter for IDC and ILC
    idc_mask = clinical_df[diagnosis_col].str.contains("ductal", case=False, na=False)
    ilc_mask = clinical_df[diagnosis_col].str.contains("lobular", case=False, na=False)

    filtered_df = clinical_df[idc_mask | ilc_mask][['cases.submitter_id', diagnosis_col]].copy()
    filtered_df['label'] = filtered_df[diagnosis_col].apply(lambda x: 'pos' if 'ductal' in x.lower() else 'neg')

    filtered_df['case_id'] = filtered_df['cases.submitter_id'].str.upper()

    return filtered_df



def select_columns_and_save(merged_df, output_path):
    selected_columns = merged_df[['Subject ID', 'cases.submitter_id', 'label', 'diagnoses.primary_diagnosis']]
    print(selected_columns.head())

    label_counts = merged_df['label'].value_counts()
    print(label_counts)

    merged_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")


def main(metadata_path, clinical_path, output_path):

    metadata_df, clinical_df = load_metadata(metadata_path, clinical_path)
    filtered_clinical_df = filter_and_label_data(clinical_df)
    merged_df = merge_data(metadata_df, filtered_clinical_df)
    select_columns_and_save(merged_df, output_path)


# metadata_file_path = '/Data/TCGA-BRCA/metadata.csv'
# clinical_file_path = '/Data/Clinical_Data/clinical.tsv'
# output_file_path = "/output/TCGA-Clinical/tcga_brca_idc_ilc_labels_combine.csv"

# main(metadata_file_path, clinical_file_path, output_file_path)

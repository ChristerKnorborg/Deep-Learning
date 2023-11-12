import csv

# Define the input and output CSV file names
input_csv = 'FINALFINAL.csv'  # Your current CSV file
output_csv = 'FINALFINALFixedCORRECTS.csv'  # The new CSV file to be created

# Initialize containers for the training and validation data
training_data = {}
validation_data = {}

# Read the data from the input CSV and separate it into training and validation
with open(input_csv, mode='r') as infile:
    reader = csv.DictReader(infile)
    for row in reader:
        epoch = row['Epoch']
        if row['Train Loss'] != '0':  # Check if 'Train Loss' is not zero, then it's training data
            training_data[epoch] = row
        else:  # Otherwise, it's validation data
            validation_data[epoch] = row

# Write the combined data to the output CSV
with open(output_csv, mode='w', newline='') as outfile:
    fieldnames = [
        'Epoch', 'Train Loss', 'Validation Loss', 'Correct Predictions', 'Localization Errors', 
        'Other Errors', 'Background Predictions', 'Total Bounding Boxes', 'TP_class_loca_good', 
        'FP_class_bad_IOU_good', 'FP_class_good_IOU_bad', 'FP_class_IOU_good', 
        'FN_class_good_IOU_bad', 'FN_class_bad_IOU_good', 'FN_class_bad_IOU_bad'
    ]
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()

    # Assuming we have the same number of epochs for both training and validation
    for epoch in sorted(training_data.keys(), key=int):
        training_row = training_data[epoch]
        validation_row = validation_data.get(epoch, {})
        
        combined_row = {field: training_row.get(field, '') for field in fieldnames}
        for field in fieldnames[2:]:  # Skip 'Epoch' and 'Train Loss' for validation data
            combined_row[field] = validation_row.get(field, combined_row[field])
        
        writer.writerow(combined_row)

print(f'New CSV file "{output_csv}" has been created with the specified metrics.')

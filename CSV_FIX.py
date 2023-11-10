import csv

# Define the input and output CSV file names
input_csv = 'model5ADAM10000.csv'  # Your current CSV file
output_csv = 'model5ADAM10000FixedCORRECTS.csv'  # The new CSV file to be created

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
    fieldnames = ['Epoch', 'Train Loss', 'Validation Loss', 'Correct Predictions', 'Localization Errors', 'Other Errors', 'Background Predictions', 'Total Bounding Boxes']
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()

    # Assuming we have the same number of epochs for both training and validation
    for epoch in sorted(training_data.keys(), key=int):
        training_row = training_data[epoch]
        validation_row = validation_data.get(epoch, {})
        combined_row = {
            'Epoch': epoch,
            'Train Loss': training_row.get('Train Loss', ''),
            'Validation Loss': validation_row.get('Validation Loss', ''),
            'Correct Predictions': validation_row.get('Correct Predictions', ''),
            'Localization Errors': validation_row.get('Localization Errors', ''),
            'Other Errors': validation_row.get('Other Errors', ''),
            'Background Predictions': validation_row.get('Background Predictions', ''),
            'Total Bounding Boxes': validation_row.get('Total Bounding Boxes', '')
        }
        writer.writerow(combined_row)

print(f'New CSV file "{output_csv}" has been created with the specified metrics.')

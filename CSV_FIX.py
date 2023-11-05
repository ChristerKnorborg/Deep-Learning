import csv

# Define the input and output CSV file names
input_csv = 'model4.csv'  # Your current CSV file
output_csv = 'model4Fixed.csv'  # The new CSV file to be created

# Initialize containers for the training and validation data
training_data = {}
validation_data = {}

# Read the data from the input CSV and separate it into training and validation
with open(input_csv, mode='r') as infile:
    reader = csv.reader(infile)
    headers = next(reader)  # Skip the header

    for i, row in enumerate(reader):
        epoch = row[0]
        if i % 2 == 0:  # Assuming even rows are training data
            training_data[epoch] = row[1]
        else:  # Assuming odd rows are validation data
            validation_data[epoch] = row[2]

# Write the combined data to the output CSV
with open(output_csv, mode='w', newline='') as outfile:
    writer = csv.writer(outfile)
    # Write the header
    writer.writerow(['Epoch', 'Train Loss', 'Validation Loss'])

    # Assuming we have the same number of epochs for both training and validation
    for epoch in sorted(training_data.keys(), key=int):
        writer.writerow([epoch, training_data[epoch], validation_data.get(epoch, '')])

print(f'New CSV file "{output_csv}" has been created with separate training and validation loss columns.')
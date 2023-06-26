from zipfile import ZipFile
file_path= "data_zip.zip"

# Extract all the contents of the zip file in current directory
with ZipFile(file_path, 'r') as zip:
    zip.printdir()
    zip.extractall()
    print('Done!')
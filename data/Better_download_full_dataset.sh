#!/bin/bash
set -e

# Function to download a file with robust options
download_file() {
    local url="$1"
    local outfile="$2"
    echo "Downloading $outfile ..."
    wget -c -t 0 --waitretry=5 --read-timeout=20 --timeout=30 "$url" -O "$outfile"
}

# Download data files from Zenodo
download_file "https://zenodo.org/records/12944960/files/2000-2002.zip?download=1" "2000-2002.zip"
download_file "https://zenodo.org/records/12945014/files/2003-2005.zip?download=1" "2003-2005.zip"
download_file "https://zenodo.org/records/12945028/files/2006-2008.zip?download=1" "2006-2008.zip"
download_file "https://zenodo.org/records/12945040/files/2009-2011.zip?download=1" "2009-2011.zip"
download_file "https://zenodo.org/records/12945050/files/2012-2014.zip?download=1" "2012-2014.zip"
download_file "https://zenodo.org/records/12945058/files/2015-2017.zip?download=1" "2015-2017.zip"
download_file "https://zenodo.org/records/12945066/files/2018-2020.zip?download=1" "2018-2020.zip"

# Unzip each downloaded file and remove the zip file afterwards
for file in *.zip; do
    echo "Unzipping $file ..."
    unzip -o "$file" -d ./
    rm "$file"
done

echo "All files downloaded and extracted."

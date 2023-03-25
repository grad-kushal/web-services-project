import json


def read_api_data(filename):
    """
    Read the API data from the file
    :param filename: The name of the file to read
    :return: The JSON object containing the API data
    """
    with open(filename, 'r', encoding="ISO-8859-1") as f:
        lines = f.readlines()

    api_records = []

    for line in lines:
        fields = line.strip().split('$#$')
        api_record = {
            'id': fields[0],
            'title': fields[1],
            'summary': fields[2],
            'rating': fields[3],
            'name': fields[4],
            'label': fields[5],
            'author': fields[6],
            'description': fields[7],
            'type': fields[8],
            'downloads': fields[9],
            'useCount': fields[10],
            'sampleUrl': fields[11],
            'downloadUrl': fields[12],
            'dateModified': fields[13],
            'remoteFeed': fields[14],
            'numComments': fields[15],
            'commentsUrl': fields[16],
            'tags': fields[17].split('###'),
            'category': fields[18],
            'protocols': fields[19],
            'serviceEndpoint': fields[20],
            'version': fields[21],
            'wsdl': fields[22],
            'dataFormats': fields[23],
            'apiGroups': fields[24],
            'example': fields[25],
            'clientInstall': fields[26],
            'authentication': fields[27],
            'ssl': fields[28],
            'readonly': fields[29],
            'VendorApiKits': fields[30],
            'CommunityApiKits': fields[31],
            'blog': fields[32],
            'forum': fields[33],
            'support': fields[34],
            'accountReq': fields[35],
            'commercial': fields[36],
            'provider': fields[37],
            'managedBy': fields[38],
            'nonCommercial': fields[39],
            'dataLicensing': fields[40],
            'fees': fields[41],
            'limits': fields[42],
            'terms': fields[43],
            'company': fields[44],
            'updated': fields[45]
        }

        api_records.append(api_record)

    return api_records


def main():
    """
    Main function
    :return: None
    """
    # Read the API data from the file
    api_records_json = read_api_data('data/api.txt')

    # Print the JSON object
    print(api_records_json)


if __name__ == '__main__':
    main()

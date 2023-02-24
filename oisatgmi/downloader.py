import os
from pathlib import Path
import sys
import json
import certifi
import urllib3
from time import sleep
import requests


def _charconv1(char):
    return("'{}'".format(char))


def _charconv2(char):
    return('"{}"'.format(char))


def _get_http_data(http1, svcurl1, request):
    hdrs = {'Content-Type': 'application/json',
            'Accept': 'application/json'}
    data = json.dumps(request)
    r = http1.request('POST', svcurl1, body=data, headers=hdrs)
    response = json.loads(r.data)
    # Check for errors
    if response['type'] == 'jsonwsp/fault':
        print('API Error: faulty request')
        sys.exit(1)
    return response


class downloader(object):
    def __init__(self, latll: float, latur: float, lonll: float, lonur: float, datestart: str, dateend: str):
        '''
            Initialize the downloader object
            Input:
                latll [float]: a latitude at the lower left corner of the region of interest           
                latur [float]: a latitude at the upper right corner of the region of interest
                lonll [float]: a longitude at the lower left corner of the region of interest
                lonur [float]: a longitude at the upper right corner of the region of interst
                datestart [str]: the start date in "YYYY-MM-DD"
                datesend  [str]: the end date in "YYYY-MM-DD"
        '''
        self.latll = latll
        self.latur = latur
        self.lonll = lonll
        self.lonur = lonur
        self.datestart = datestart
        self.dateend = dateend

    def download_tropomi_l2(self, product_tag: str, output_fld: Path, maxpage=30, username="s5pguest", password="s5pguest"):
        '''
            download the tropomi data
            Inputs:
                product_tag [int]: 1 -> NO2
                               2 -> HCHO
                               3 -> CH4
                               4 -> CO
                output_fld [Path]: a pathlib object describing the output folder
                maxpage [int]: the number of pages in the xml file
                username [str]: the username to log on s5phub
                password [str]: the password to log on s5phub 
        '''
        # define product string
        if product_tag == 1:
            product_name = "%5F%5FNO2%5F%5F%5F"
        if product_tag == 2:
            product_name = "%5F%5FHCHO%5F%5F"
        if product_tag == 3:
            product_name = "%5F%5FCH4%5F%5F%5F"
        if product_tag == 4:
            product_name = "%5F%5FCO%5F%5F%5F%5F"
        # loop over the pages
        for page in range(0, maxpage):
            searcher = "https://s5phub.copernicus.eu/dhus/search?start="
            searcher += f"{0+page*100:01}" + "&rows=100&q=footprint:%22"
            searcher += "Intersects(POLYGON((" + f"{self.lonll:.4f}" + "%20"
            searcher += f"{self.latll:.4f}" + "," + f"{self.lonur:.4f}" + "%20"
            searcher += f"{self.latll:.4f}" + "," + \
                f"{self.lonur:.4f}" + "%20" + f"{self.latur:.4f}"
            searcher += "," + f"{self.lonll:.4f}" + \
                "%20" + f"{self.latur:.4f}" + ","
            searcher += f"{self.lonll:.4f}" + "%20" f"{self.latll:.4f}" + \
                ")))%22%20AND%20(%20beginPosition:"
            searcher += "%5B" + self.datestart + "T00:00:00.000Z%20TO%20" + self.dateend
            searcher += "T23:59:59.999Z%5D%20AND%20endPosition:%5B" + self.datestart
            searcher += "T00:00:00.000Z%20TO%20" + self.dateend + "T23:59:59.999Z%5D%20)%"
            searcher += "20AND%20((platformname:Sentinel-5)%20AND%20(producttype:"
            searcher += "L2"f"{product_name}""%20AND%20processinglevel:L2))"

            # retrieve the product names and save them in temp/tropomi.xml
            if not os.path.exists('temp'):
                os.makedirs('temp')
            cmd = ' curl -L -k -u s5pguest:s5pguest "' + searcher + '"> temp/tropomi.xml'
            os.system(cmd)

            # read the xml file
            with open('temp/tropomi.xml', 'r') as f:
                data = f.read()
                data1 = data.split('<str name="uuid">')

                # list the files to be downloaded in this particular page
                list_file = []

                if len(data1) == 0:
                    break

                for i in range(0, len(data1)-1):
                    list_file.append(data1[i+1].split('</str>')[0])

                # download the data
                for fname in list_file:
                    cmd = "wget -nH -nc --no-check-certificate --content-disposition --continue "
                    cmd += "--user s5pguest --password s5pguest "
                    cmd += _charconv2("https://s5phub.copernicus.eu/dhus/odata/v1/Products(" +
                                      _charconv1(fname) + ")/\$value")
                    cmd += " -P" + (output_fld.as_posix()) + "/"
                    if not os.path.exists(output_fld.as_posix()):
                        os.makedirs(output_fld.as_posix())
                    os.system(cmd)

    def download_omi_l2(self, product_tag: int, output_fld: Path, product_name=None, username=None, password=None):
        '''
            download the omi data
            Inputs:
                product_tag [int]: 1 -> NO2
                                   2 -> HCHO (not implemented waiting for the newest version)
                output_fld [Path]: a pathlib object describing the output folder
                product_name [str] (optional): a product name to overwrite product_tag default values
                username [str] (optional): the username to log on nasa gesdisc
                password [str] (optional): the password to log on nasa gesdisc
        '''
        # this is based on the instruction presented at NASA GES DISC
        # if username and password are set:
        if (username is not None) and (password is not None):
            cmd = "touch ~/.netrc"
            os.system(cmd)
            cmd = "echo " + '"' + "machine urs.earthdata.nasa.gov login " + username + " password " +\
                password + '"' + " >> ~/.netrc"
            os.system(cmd)
            cmd = "chmod 0600 ~/.netrc"
            os.system(cmd)
            cmd = "touch ~/.urs_cookies"
            os.system
        # Create a PoolManager instance to make requests.
        http = urllib3.PoolManager(
            cert_reqs='CERT_REQUIRED', ca_certs=certifi.where())
        # Set the URL for the GES DISC API endpoint for dataset searches
        svcurl = 'https://disc.gsfc.nasa.gov/service/subset/jsonwsp'
        # the product target
        if product_tag == 1:
            product = 'OMI_MINDS_NO2_1.1'
        elif product_tag == 2:
            product = 'OMIHCHO'
        if (product_name is not None):
            prouct = product_name
        # Set up the JSON WSP request for API method: subset
        subset_request = {
            'methodname': 'subset',
            'type': 'jsonwsp/request',
            'version': '1.0',
            'args': {'role': 'subset', 'start': self.datestart + 'T00:00:00.000Z',
                     'end': self.dateend + 'T23:59:59.999Z',
                     'box': [self.lonll, self.latll,
                             self.lonur, self.latur], 'data': [{'datasetId': product}]}
        }
        response = _get_http_data(http, svcurl, subset_request)
        myJobId = response['result']['jobId']
        # Construct JSON WSP request for API method: GetStatus
        status_request = {
            'methodname': 'GetStatus',
            'version': '1.0',
            'type': 'jsonwsp/request',
            'args': {'jobId': myJobId}
        }
        # Check on the job status after a brief nap
        while response['result']['Status'] in ['Accepted', 'Running']:
            sleep(5)
            response = _get_http_data(http, svcurl, status_request)
            status = response['result']['Status']
            percent = response['result']['PercentCompleted']
            print('Job status: %s (%d%c complete)' %
                  (status, percent, '%'))
            if response['result']['Status'] == 'Succeeded':
                print('Job Finished:  %s' % response['result']['message'])
                # Retrieve a plain-text list of results in a single shot using the saved JobID
                result = requests.get(
                    'https://disc.gsfc.nasa.gov/api/jobs/results/'+myJobId)
                try:
                    result.raise_for_status()
                    urls = result.text.split('\n')
                    for url in urls:
                        cmd = "wget -nH -nc --no-check-certificate --content-disposition --continue "
                        cmd += "--load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --auth-no-challenge=on "
                        cmd += "--keep-session-cookies "
                        cmd += '"' + str(url)[:-1] + '"'
                        cmd += " -P " + (output_fld.as_posix())
                        if not os.path.exists(output_fld.as_posix()):
                            os.makedirs(output_fld.as_posix())
                        os.system(cmd)
                except:
                    print('Request returned error code %d' %
                          result.status_code)
            else:
                continue


# testing
if __name__ == "__main__":

    dl_obj = downloader(37, 40, -79, -73.97, '2019-05-01', '2019-06-01')
    dl_obj.download_tropomi_l2(1, Path('download_bucket/no2/'))
    #dl_obj.download_omi_l2(1, Path('download_bucket/omi_no2/'))

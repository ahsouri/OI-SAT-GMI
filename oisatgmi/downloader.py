import os
from pathlib import Path
import sys
import json
import certifi
import urllib3
from time import sleep
import requests
import datetime
import requests
from bs4 import BeautifulSoup


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


def _daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + datetime.timedelta(n)


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

    def download_tropomi_l2_old(self, product_tag: str, output_fld: Path, maxpage=3000, username="s5pguest", password="s5pguest"):
        '''
            download the tropomi data
            Inputs:
                product_tag [str]: 1 -> NO2
                               2 -> HCHO
                               3 -> CH4
                               4 -> CO
                output_fld [Path]: a pathlib object describing the output folder
                maxpage [int]: the number of pages in the xml file
                username [str]: the username to log on s5phub
                password [str]: the password to log on s5phub 
        '''
        # define product string
        if product_tag == 'NO2':
            product_name = "%5F%5FNO2%5F%5F%5F"
        if product_tag == 'HCHO':
            product_name = "%5F%5FHCHO%5F%5F"
        if product_tag == 'CH4':
            product_name = "%5F%5FCH4%5F%5F%5F"
        if product_tag == 'CO':
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
                    cmd = "wget --header " + _charconv2("Authorization: Bearer $ACCESS_TOKEN") + "  "
                    cmd += "--wait=100 --random-wait --content-disposition --continue "
                    #cmd += "--user=s5pguest --password=s5pguest "
                    cmd += _charconv2("http://catalogue.dataspace.copernicus.eu/odata/v1/Products(" +
                                      fname + ")/\$value")
                    cmd += " -P " + (output_fld.as_posix()) + "/"
                    if not os.path.exists(output_fld.as_posix()):
                        os.makedirs(output_fld.as_posix())
                    sleep(5.0)
                    print(cmd)
                    os.system(cmd)

    def download_tropomi_l2(self, product_tag: str, output_fld: Path, product_name=None, username=None, password=None):
        '''
            download the tropomi data from NASA GES DISC
            Inputs:
                product_tag [str]: NO2
                                   HCHO
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
            os.system(cmd)
        # Create a PoolManager instance to make requests.
        http = urllib3.PoolManager(
            cert_reqs='CERT_REQUIRED', ca_certs=certifi.where())
        # Set the URL for the GES DISC API endpoint for dataset searches
        svcurl = 'https://disc.gsfc.nasa.gov/service/subset/jsonwsp'
        # the product target
        if product_tag == 'NO2':
            product = 'S5P_L2__NO2____HiR_2'
        elif product_tag == 'HCHO':
            product = 'S5P_L2__HCHO___HiR_2'
        if (product_name is not None):
            product = product_name
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
                        cmd = "wget "
                        cmd += "--continue --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --auth-no-challenge=on "
                        cmd += "--keep-session-cookies --timeout=600 "
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
            
    def download_omi_l2(self, product_tag: str, output_fld: Path, product_name=None, username=None, password=None):
        '''
            download the omi data
            Inputs:
                product_tag [str]: NO2
                                   HCHO (not implemented waiting for the newest version)
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
            os.system(cmd)
        # Create a PoolManager instance to make requests.
        http = urllib3.PoolManager(
            cert_reqs='CERT_REQUIRED', ca_certs=certifi.where())
        # Set the URL for the GES DISC API endpoint for dataset searches
        svcurl = 'https://disc.gsfc.nasa.gov/service/subset/jsonwsp'
        # the product target
        if product_tag == 'NO2':
            product = 'OMI_MINDS_NO2_1.1'
        elif product_tag == 'HCHO':
            product = 'OMHCHO_003'
        elif product_tag == 'O3':
            #product = 'OMDOAO3_003'
            product = 'OMTO3_003'
        if (product_name is not None):
            product = product_name
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
                        cmd = "wget -nH -nc --no-check-certificate "
                        if product_tag != 'O3':
                            cmd += "--content-disposition "
                        cmd += "--continue --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --auth-no-challenge=on "
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

    def download_mopitt_l2(self,  output_fld: Path):
        '''
            download the MOPITT CO L3 observations
            output_fld [Path]: a pathlib object describing the output folder
        '''
        # convert dates to datetime
        start_date = datetime.date(int(self.datestart[0:4]), int(
            self.datestart[5:7]), int(self.datestart[8:10]))
        end_date = datetime.date(int(self.dateend[0:4]), int(
            self.dateend[5:7]), int(self.dateend[8:10]))

        for single_date in _daterange(start_date, end_date):
            url = 'https://opendap.larc.nasa.gov/opendap/MOPITT/MOP03J.009/'
            url += str(single_date.year) + '.'
            url += f"{single_date.month:02}" + '.'
            url += f"{single_date.day:02}" + '/'

            reqs = requests.get(url)
            soup = BeautifulSoup(reqs.text, 'html.parser')
            for link in soup.find_all('a'):
                if (link.get('href')[0:6] != "MOP03J") or (link.get('href')[-3::] != "he5"):
                    continue
                # print(link.get('href'))
                cmd = "wget -nH -nc --no-check-certificate  --continue "
                cmd += '"' + url + link.get('href') + '"'
                cmd += " -P " + (output_fld.as_posix())
                print(cmd)
                if not os.path.exists(output_fld.as_posix()):
                    os.makedirs(output_fld.as_posix())
                os.system(cmd)

    def merra2_gmi(self, output_fld: Path):
        '''
            download the merra2-gmi data
            output_fld [Path]: a pathlib object describing the output folder
        '''
        # convert dates to datetime
        start_date = datetime.date(int(self.datestart[0:4]), int(
            self.datestart[5:7]), int(self.datestart[8:10]))
        end_date = datetime.date(int(self.dateend[0:4]), int(
            self.dateend[5:7]), int(self.dateend[8:10]))

        for single_date in _daterange(start_date, end_date):

            url = 'https://portal.nccs.nasa.gov/datashare/merra2_gmi/Y'
            url += str(single_date.year) + '/M'
            url += f"{single_date.month:02}" + '/MERRA2_GMI.tavg3_3d_tac_Nv.'
            url += str(single_date.year) + \
                f"{single_date.month:02}" + f"{single_date.day:02}"
            url += '.nc4'
            cmd = "wget -nH -nc --no-check-certificate --content-disposition --continue "
            cmd += '"' + url + '"'
            cmd += " -P " + (output_fld.as_posix())
            if not os.path.exists(output_fld.as_posix()):
                os.makedirs(output_fld.as_posix())
            os.system(cmd)

            url = 'https://portal.nccs.nasa.gov/datashare/merra2_gmi/Y'
            url += str(single_date.year) + '/M'
            url += f"{single_date.month:02}" + '/MERRA2_GMI.tavg3_3d_met_Nv.'
            url += str(single_date.year) + \
                f"{single_date.month:02}" + f"{single_date.day:02}"
            url += '.nc4'
            cmd = "wget -nH -nc --no-check-certificate --content-disposition --continue "
            cmd += '"' + url + '"'
            cmd += " -P " + (output_fld.as_posix())
            if not os.path.exists(output_fld.as_posix()):
                os.makedirs(output_fld.as_posix())
            os.system(cmd)

    def omi_hcho_cfa(self, output_fld: Path):
        '''
            download the omi SAO HCHO observations
            output_fld [Path]: a pathlib object describing the output folder
        '''
        # convert dates to datetime
        start_date = datetime.date(int(self.datestart[0:4]), int(
            self.datestart[5:7]), int(self.datestart[8:10]))
        end_date = datetime.date(int(self.dateend[0:4]), int(
            self.dateend[5:7]), int(self.dateend[8:10]))

        for single_date in _daterange(start_date, end_date):

            url = 'https://waps.cfa.harvard.edu/sao_atmos/data/omi_hcho/OMI-HCHO-L2/'
            url += str(single_date.year) + '/'
            url += f"{single_date.month:02}" + '/'
            url += f"{single_date.day:02}" + '/'

            reqs = requests.get(url)
            soup = BeautifulSoup(reqs.text, 'html.parser')
            for link in soup.find_all('a'):
                print(link.get('href'))
                cmd = "wget -nH -nc --no-check-certificate --content-disposition --continue "
                cmd += '"' + url + link.get('href') + '"'
                cmd += " -P " + (output_fld.as_posix())
                if not os.path.exists(output_fld.as_posix()):
                    os.makedirs(output_fld.as_posix())
                os.system(cmd)


# testing
if __name__ == "__main__":

    dl_obj = downloader(19, 61, -136, -54, '2019-05-01', '2019-05-01')
    #dl_obj = downloader(-90, 90, -180, 180, '2005-06-01', '2005-07-01')
    dl_obj.download_tropomi_l2('HCHO', Path('download_bucket/trop_hcho/'))
    #dl_obj.download_omi_l2('HCHO', Path('download_bucket/omi_no2/'))
    #dl_obj.omi_hcho_cfa( Path('download_bucket/omi_hcho_PO3/'))
    #dl_obj.download_omi_l2('O3', Path('download_bucket/omi_o3/'))
    #dl_obj.download_mopitt_l2(Path('download_bucket/mopitt_CO/'))
    # dl_obj.merra2_gmi(Path('download_bucket/gmi/'))

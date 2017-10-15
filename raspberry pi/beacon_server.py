# coding:utf-8

import sys,os,random
import subprocess
import signal
from time import sleep
from BaseHTTPServer import HTTPServer, BaseHTTPRequestHandler
import threading
import urlparse

HOST = "localhost"
PORT = 8888


class MyHTTPServer(HTTPServer):
    proc = None
    prev_path = None
    def go_iBeacon(self,label,strength):
        cmd = "ibeacon 200 e2c56db5dffb48d2b060d0f5a71096e0 {} {} -59".format(label,strength)
        self.proc = subprocess.Popen(cmd.strip().split(" "))

    

class MyHTTPRequestHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        parsed_path = urlparse.urlparse(self.path)
        path = parsed_path.path.replace('/', '')
        l_s = path.split('_')
        label = l_s[0]
        strength = l_s[1]

        if label in '012' and strength in '01234':
            if self.server.prev_path != path:
                self.server.prev_path = path

                if self.server.proc is not None:
                    #print self.server.proc
                    self.server.proc.terminate()
                    self.server.proc.wait()
                    self.server.go_iBeacon(label,strength)
                else:
                    print "newly generated"
                    self.server.go_iBeacon(label,strength)
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=UTF-8')
        self.end_headers()
        return





if __name__ == "__main__":
    sv = MyHTTPServer((HOST, PORT), MyHTTPRequestHandler)
    try:
        sv.serve_forever()
    except KeyboardInterrupt:
        print "server terminated"
   #sv = BeaconSwitcher()
   #sv.setUp()

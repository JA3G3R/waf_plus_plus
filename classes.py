from xmlrpc.client import FastMarshaller
import pandas as pd
import socket 
import ssl
import threading
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

class HTTPServer:

    """

        This class will be used to create and start an HTTP Server which will handle the requests from client.
          
    """

    def __init__(self,address,port):
        self.port = port
        self.address=address
        self.server_sock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.waf = WAF()
        self.threads= []

        # HTTPS
        if port == 31337:
            self.context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            self.context.load_cert_chain("certs/server-cert.pem","certs/server-key.pem")
            with self.context.wrap_socket(self.server_sock,server_side=True) as ssock:
                ssock.bind((self.address,self.port))
                self.listen(ssock)
        else:
            self.server_sock.bind((self.address,self.port))
            self.listen(self.server_sock)

    def listen(self,sock):

        while True:
            sock.listen(10)
            client_sock,client_addr = sock.accept()
            # self.remove_threads()
            self.threads.append(threading.Thread(target=self.handleClient,args=(client_sock,client_addr,)))
            self.threads[len(self.threads)-1].start() 

    def remove_threads(self):
        
        dead =[]
        for i in range(len(self.threads)):
            if not self.threads[i].is_alive():
                dead.append(i)
        for i in dead:
            self.threads = self.threads[:i]+self.threads[i+1:]

    def handleClient(self,client_sock,client_addr):
        host,port = client_addr
        print("[+]DEBUG : Received connection from ",host,port)
        c=0
        while True:
            try:
                if client_sock.fileno == -1:
                    print("Closed connection")
                    return 
                req = client_sock.recv(1024).decode()
                if req:
                    print(f"RECVD: from {host}:{port}\n{req} {'-'*20}")
                    resp=self.handleRequest(req)
                    client_sock.send(resp.encode())
                else:
                    # if c==10:
                    client_sock.sendall(b'HTTP/1.1 200 OK\r\n')
                    client_sock.close()
                    return
                    c=0
                    # c+=1
            except ConnectionResetError:
                break
        return                       

    def handleRequest(self,req):
        if "\r\n\r\n" in req:
            pre,body = req.split("\r\n\r\n")
            first_line= pre.split("\r\n")[0]
            header_list = pre.split("\r\n")[1:]
        else:
            return ""

        # print(f"BODY: \n{body}\n"+"-"*30+"\n"+f"REQUEST:\n{first_line}\n"+"-"*30+"\n"+f"HEADERS:\n{header_list}\n"+"-"*30+"\n")
        
        method,url,version = first_line.split(' ')
        resp=""
        if version != "HTTP/1.1" and version != "HTTP/2":
            return "Invalid HTTP version"
        else:

            if "GET" == method:
                resp = self.doGet(url,header_list,body)

            elif "POST" == method:
                resp = self.doPost(url,header_list,body)
        return resp
                

    def doGet(self,uri,header_list,body):
        resp = "HTTP/1.1 400 Bad Request\r\n\r\nThe Client sent a malformed request is all we know."

        if self.waf.checkGet(uri,header_list):
            resource = uri.split('/?')[0]
            if not resource or resource=='/':
                uri="index.html"
            else:
                resp ="HTTP/1.1 200 OK\r\n\r\nYou passed the security check!!!"

        print(f"[+]DEBUG: REQUESTED URL : {uri}")
        return resp

    def doPost(self,url,header_list,body):
        resp = "HTTP/1.1 400 Bad Request\r\n\r\nThe Client sent a malformed request is all we know."
        if self.waf.checkPost(header_list,body):
            resp= "HTTP/1.1 200 OK\r\n\r\nYou passed the security check!!!"
        
        return resp


class WAF:

    def __init__(self,):
        self.csic_ecml_get_model = pickle.load(open('models/normal_svc_get.sav','rb'))
        self.csic_ecml_post_model = pickle.load(open('models/normal_svc_post.sav','rb'))
        self.tfidf_get = pickle.load(open('models/tfidf_get.sav','rb'))
        self.tfidf_post = pickle.load(open('models/tfidf_post.sav','rb'))

    def make_request_features(self,header_list,method,data="",query=""):
        
        ngrams = [' ', '!', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.',
        '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<',
        '=', '>', '?', '@', '[', '\\', '\]', '_', '`', 'a', 'b', 'c', 'd', 'e',
        'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
        't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~']

        vocabulary = {}
        for i in range(len(ngrams)):
            vocabulary[ngrams[i]] = i

        request=""
        for i in header_list:
            key,value = i.split(':')[:2]
            key=key.strip(' ').lower()
            value = value.strip(' ').lower()
            if key == 'connection':
                request+=value
            elif key == 'accept':
                request+=value
            elif key == 'accept-charset':
                request += value
            elif key == 'accept-language':
                request += value
            elif key == 'cache-control':
                request += value
            elif key == 'pragma':
                request+=value
            elif key == 'user-agent':
                request += value
            elif (key == 'content-type' and method == 'POST'):
                request += value

        if (len(data) and method == 'POST'):
            request += data
        if(len(query) and method == 'GET'):
            request += query
        if method=='GET':
            request = pd.DataFrame(self.tfidf_get.transform([request]).todense(),columns=self.tfidf_get.get_feature_names_out())
        elif method == 'POST':
            request = pd.DataFrame(self.tfidf_post.transform([request]).todense(),columns=self.tfidf_post.get_feature_names_out())

        return request


    def checkGet(self,url,header_list):
        request = self.make_request_features(header_list,method='GET',query=url)
        # prediction = self.xss_check.predict(request)[0]
        # if prediction==1:
        #     print('XSS Detected!!')
        #     return False
        prediction = self.csic_ecml_get_model.predict(request)[0]
        print('Get Model Predicted: ',prediction)
        if prediction==0:
            return False
        else:
            return True


    def checkPost(self,header_list,data):
        request = self.make_request_features(header_list,method='POST',data=data)
        print()
        prediction = self.csic_ecml_post_model.predict(request)[0]
        print("Post Model Predicted: ",prediction)
        if prediction==0:
            return False
        else:
            return True
        

if __name__ =="__main__":
    # print(help(HTTPServer))
    HTTPServer("0.0.0.0",int(input("Enter port number: ")))

import socket 
import ssl
import threading

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
        
        while True:
            try:
                if client_sock.fileno == -1:
                    print("Closed connection")
                    return 
                req = client_sock.recv(1024).decode()
                print(f"RECVD: from {host}:{port}\n{req} {'-'*20}")
                resp=self.handleRequest(req)
                
                client_sock.send(resp.encode())
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
        
        if version != "HTTP/1.1" and version != "HTTP/2":
            return "Invalid HTTP version"
        else:

            if "GET" == method:
                resp = self.doGet(url,header_list,body)

            elif "POST" == method:
                resp = self.doPost(url,header_list,body)
                
        return resp
                

    def doGet(self,url,header_list,body):
        resp = "HTTP/1.1 400 Bad Request\r\n\r\nThe Client sent a malformed request is all we know."
        resource = url.split('/')[1]
        if self.waf.checkGet(url,header_list,body):
            resp = "HTTP/1.1 200 OK\r\nContent-Encoding: text/html\r\n\r\n"
            with open(resource,"r") as f:
                resp += f.read()
            # resp  = "HTTP/1.1 200 OK\r\n\r\nYou passed the security check!!!"
        print(f"[+]DEBUG: REQUESTED URL : {url}")

        return resp

    def doPost(self,url,header_list,body):
        resp = "HTTP/1.1 400 Bad Request\r\n\r\nThe Client sent a malformed request is all we know."
        if self.waf.checkPost(url,header_list,body):
            resp= "HTTP/1.1 200 OK\r\n\r\nYou passed the security check!!!"
        return resp


class WAF:

    def __init__(self,):
        pass    

    def checkGet(self,url,header_list,body):
        return False

    def checkPost(self,req):
        return True

if __name__ =="__main__":
    # print(help(HTTPServer))
    HTTPServer("0.0.0.0",int(input("Enter port number: ")))
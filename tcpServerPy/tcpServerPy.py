"""! Prototype TCP server script created by PeterC - 15-06-2024"""
# NOTE: the current implementation allows one request at a time, since it is thought for the evaluation of torch models in MATLAB.

# Python imports
import socketserver
import pickle
import os

# Request handler class 
class pytcp_requestHandler(socketserver.BaseRequestHandler):
    '''Request Handler class for tcp server'''
    def handle(self) -> None:
        '''Handle method'''
        print(f"Connected client: {self.client_address}")
        return super().handle()

# Define tcp server class
class pytcp_server(socketserver.TCPServer):
    '''Python-based custom tcp server class using socketserver module'''
    def __init__(self, serverAddress:str, requestHandler:pytcp_requestHandler) -> None:
        '''__init__ method from super class'''
        super.__init__(serverAddress)


# Custom request handler for torch models evaluation
class pytcp_requestHandler_torchEval(pytcp_requestHandler):
    '''Request Handler class for tcp server tailored to execute torch model evaluation using data from client'''
    def handle(self):
        '''Client data processing request handler'''
        print('TODO')



# MAIN SCRIPT
def main():
    print('Main in this script does nothing...')

if __name__ == "__main__":
    main()

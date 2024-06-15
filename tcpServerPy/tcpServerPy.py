"""! Prototype TCP server script created by PeterC - 15-06-2024"""
# NOTE: the current implementation allows one request at a time, since it is thought for the evaluation of torch models in MATLAB.

# Python imports
import socketserver
import pickle
import os

# %% Data processing function wrapper as generic interface in RequestHandler for TCP servers - PeterC - 15-06-2024
class DataProcessor():
    '''Data processing function wrapper as generic interface in RequestHandler for TCP servers'''
    def __init__(self, processDataFcn:callable, inputTargetType):
        '''Constructor'''
        self.processDataFcn = processDataFcn
        self.inputTargetType = inputTargetType

    def process(self, inputData):
        '''Processing method running specified processing function'''
        return self.processDataFcn(inputData)
    
    def convert(self, inputData):
        '''Data conversion function from raw data array to specified target type'''
        if not isinstance(inputData, self.inputTargetType):
            try:
                inputData = self.inputTargetType(inputData)
            except Exception as errMsg:
                raise TypeError('Data conversion from raw data array to specified target type {targetType} failed with error: \n'.format(self.inputTargetType) + str(errMsg))


# %% Request handler class - PeterC + GPT4o- 15-06-2024
class pytcp_requestHandler(socketserver.BaseRequestHandler):
    '''Request Handler class for tcp server'''
    def __init__(self, request, client_address, server, DataProcessor:DataProcessor):
        ''''Constructor'''
        self.DataProcessor = DataProcessor # Initialize DataProcessing object for handle
        self.BufferSizeInBytes = DataProcessor.BufferSizeInBytes
        super().__init__(request, client_address, server)

    def handle(self) -> None:
        '''Handle method'''
        print(f"Handling request from client: {self.client_address}")
        try:
            while True:
                # Read the length of the data (4 bytes) specified by the client
                # bufferSizeFromClient = self.request.recv(4)
                # if not bufferSizeFromClient:
                #     break
                # bufferSize = int.from_bytes(bufferSizeFromClient, 'big')
                bufferSize = self.BufferSizeInBytes

                # Read the entire data buffer
                dataBuffer = b''
                while len(dataBuffer) < bufferSize:
                    packet = self.request.recv(bufferSize - len(dataBuffer))
                    if not packet:
                        break
                    dataBuffer += packet

                # Deserialize the received data buffer using pickle
                dataArray = pickle.loads(dataBuffer)
                print(f"Received array:\t{dataArray}")

                # Move the data to DataProcessor and process according to specified function
                outputData = self.DataProcessor(dataArray)

                # Serialize the output data before sending them back
                outputDataSerialized = pickle.dumps(outputData) 
                #outputDataSizeInBytes = len(outputDataSerialized)

                # Send the length of the processed data - CURRENTLY NOT IN USE 
                # self.request.sendall(outputDataSizeInBytes.to_bytes(4, 'big'))

                # Send the serialized output data
                self.request.sendall(outputDataSerialized)

        except Exception as e:
            print(f"Error occurred while handling request: {e}")
        finally:
            print(f"Connection with {self.client_address} closed")


# %% TCP server class - PeterC -15-06-2024
class pytcp_server(socketserver.TCPServer):
    '''Python-based custom tcp server class using socketserver module'''
    def __init__(self, serverAddress: tuple[str|bytes|bytearray, int], RequestHandlerClass: pytcp_requestHandler, bindAndActivate:bool=True) -> None:
        '''Constructor for custom tcp server'''
        self.DataProcessor = DataProcessor # Initialize DataProcessing object for handle
        super().__init__(serverAddress, RequestHandlerClass, bindAndActivate)
    
    def finish_request(self, request, client_address) -> None:
        '''Function evaluating Request Handler'''
        self.RequestHandlerClass(request, client_address, self, self.DataProcessor)

# %% MAIN SCRIPT
def main():
    print('Main in this script does nothing...')
if __name__ == "__main__":
    main()

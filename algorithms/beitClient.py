import base64
import grpc
from grpc_protos import beit3service_pb2
from grpc_protos import beit3service_pb2_grpc

def send_request_to_server(tensor_data, frame_id, tracker_id):
    # Set up the gRPC channel and stub
    channel = grpc.insecure_channel('127.0.0.1:3000')
    stub = beit3service_pb2_grpc.Beit3ServiceStub(channel)
    
    # Base64 encode the tensor data
    # encoded_tensor_data = base64.b64encode(tensor_data).decode('utf-8')
    
    # Create the request
    request = beit3service_pb2.Beit3Request(
        tensor=tensor_data,
        frame_id=frame_id,
        tracker_id=tracker_id
    )
    
    # Send the request and get the response
    response = stub.EnqueueItem(request)
    
    return response.retcode

if __name__ == "__main__":
    # You need to have your tensor data in a binary format to encode
    # For this example, let's assume it's a binary string.
    # In practice, you'd probably load and serialize your tensor data.
    TENSOR_DATA = b"YOUR_BINARY_TENSOR_DATA"
    
    FRAME_ID = "YOUR_FRAME_ID"
    TRACKER_ID = "YOUR_TRACKER_ID"

    retcode = send_request_to_server(TENSOR_DATA, FRAME_ID, TRACKER_ID)
    print(f"Server response: {retcode}")

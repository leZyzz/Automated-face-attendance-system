import torch
from facenet_pytorch import InceptionResnetV1

def export_facenet_to_onnx():
    print("Loading PyTorch model...")
    # Load the model with pre-trained weights. Make sure to use the same 
    # weights you used for your database (usually 'vggface2' or 'casia-webface')
    # .eval() is crucial: it switches the model from training mode to inference mode!
    resnet = InceptionResnetV1(pretrained='vggface2').eval()

    # Create a dummy input tensor. 
    # Shape: (Batch Size, Channels, Height, Width) -> (1, 3, 160, 160)
    dummy_input = torch.randn(1, 3, 160, 160)

    onnx_file_path = "facenet.onnx"
    
    print("Exporting to ONNX...")
    torch.onnx.export(
        resnet,                      # The PyTorch model
        dummy_input,                 # The fake input to trace the graph
        onnx_file_path,              # Where to save the file
        export_params=True,          # Store the trained weights inside the file
        opset_version=11,            # The ONNX standard version (11 is very stable)
        do_constant_folding=True,    # Optimizes the graph by pre-calculating constants
        input_names=['input'],       # Name of the input layer
        output_names=['output'],     # Name of the output layer (the 512D embedding)
        
        # This allows the model to process multiple faces at once later if you want
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}} 
    )

    print(f"Export complete! Your optimized model is saved as: {onnx_file_path}")

if __name__ == "__main__":
    export_facenet_to_onnx()
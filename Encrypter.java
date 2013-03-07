public class Encrypter {
    
    public Encrypter(String keyString,
            java.io.InputStream inputStream,
            java.io.OutputStream outputStream)
            throws java.io.IOException {
        
        byte [] key = keyString.getBytes();
        byte [] buffer = new byte[256];
        int keyIndex = 0;
        int bytesRead = 0;

        while((bytesRead = inputStream.read(buffer)) != -1) {
            
            for(int i = 0; i < bytesRead; i++) {
                
                buffer[i] = (byte)(buffer[i] ^ key[keyIndex]);
                keyIndex = (keyIndex+1) % key.length;
            }
            
            outputStream.write(buffer, 0, bytesRead);
        }
    }
    
    public static void main(String[] args) {
        
        try {
            
            new Encrypter(args[0], System.in, System.out);
        } catch(java.io.IOException e) { }
    }
}

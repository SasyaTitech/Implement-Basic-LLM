
def generate_encodings(input_string, encoding_type='utf-8'):

    print("-"*100)
    print(f"{encoding_type} Examples")

    ## UTF-8 Encoding Assignment example
    utf8_encoded = input_string.encode(encoding_type)
    print("string utf-8 encoded: ", utf8_encoded)
    print("type: ", type(utf8_encoded))

    # byte values for encoded strings
    print("byte values encoded strings: ", list(utf8_encoded))

    print("-"*100)
    return utf8_encoded

def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

## UTF-16 examples
if __name__ == "__main__":
    test_string = "hello! こんにちは!"
    generate_encodings(test_string)

    generate_encodings(test_string, 'utf-16')

    generate_encodings(test_string, 'utf-32')
    print(decode_utf8_bytes_to_str_wrong("hello".encode("utf-8")))

    print(decode_utf8_bytes_to_str_wrong(test_string))


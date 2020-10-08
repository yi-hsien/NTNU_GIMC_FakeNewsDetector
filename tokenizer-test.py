## best example of all
test = '\udce9\udca7\udc90'

print("original length")
print(len(test))


test = test.encode('utf-8', errors='surrogateescape').decode('utf-8')

print(test)
print(len(test))















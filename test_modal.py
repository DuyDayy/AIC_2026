import modal
app = modal.App("test-hello")
@app.function()
def hello():
    print("Hello from modal!")
    return 42

if __name__ == "__main__":
    with app.run():
        print(hello.remote())

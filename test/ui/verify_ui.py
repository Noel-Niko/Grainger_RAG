import streamlit as st


def main():
    st.title("Simple Streamlit App")

    # Display a text input box
    user_input = st.text_input("Enter something:")

    # Display a button
    if st.button("Submit"):
        # Display the input text
        st.write("You entered:", user_input)


if __name__ == "__main__":
    main()

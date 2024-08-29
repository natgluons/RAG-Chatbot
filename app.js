document.getElementById('sendButton').addEventListener('click', async () => {
    const userInput = document.getElementById('userInput').value;
    if (userInput.trim() === '') return;

    const chatbox = document.getElementById('chatbox');
    chatbox.innerHTML += `<p>You: ${userInput}</p>`;

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: userInput }),
        });

        if (response.ok) {
            const data = await response.json();
            chatbox.innerHTML += `<p>Chatbot: ${data.response}</p>`;
        } else {
            chatbox.innerHTML += `<p>Chatbot: Error ${response.status}</p>`;
        }
    } catch (error) {
        chatbox.innerHTML += `<p>Chatbot: Network error</p>`;
    }

    document.getElementById('userInput').value = '';
    chatbox.scrollTop = chatbox.scrollHeight;
});

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

            // Add the chatbot's response
            chatbox.innerHTML += `<p>Chatbot: ${data.response}</p>`;

            // If there are sources, add them
            if (data.sources && data.sources.length > 0) {
                let sourcesHtml = '<p class="sources">Sources:<br>';
                data.sources.forEach(source => {
                    sourcesHtml += `- ${source.title} (Page ${source.page_number})<br>`;
                });
                sourcesHtml += '</p>';
                chatbox.innerHTML += sourcesHtml;
            }
        } else {
            chatbox.innerHTML += `<p>Chatbot: Error ${response.status}</p>`;
        }
    } catch (error) {
        console.error('Error:', error);
        chatbox.innerHTML += `<p>Chatbot: Network error</p>`;
    }

    document.getElementById('userInput').value = '';
    chatbox.scrollTop = chatbox.scrollHeight;
});
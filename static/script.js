let useModel;

// Load Universal Sentence Encoder properly
async function loadUSEModel() {
  useModel = await use.load();
  console.log("[INFO] USE model loaded");
}
loadUSEModel();

async function sendMessage() {
  const input = document.getElementById("user-input");
  const chatBox = document.getElementById("chat-box");
  const userText = input.value.trim();
  if (!userText || !useModel) return;

  // Add user message
  const userMsg = document.createElement("div");
  userMsg.className = "chat-message user";
  userMsg.textContent = userText;
  chatBox.appendChild(userMsg);

  // Add loading bot message
  const botMsg = document.createElement("div");
  botMsg.className = "chat-message bot";
  botMsg.textContent = "Typing...";
  chatBox.appendChild(botMsg);
  chatBox.scrollTop = chatBox.scrollHeight;

  try {
    // Generate embedding in-browser
    const embeddings = await useModel.embed([userText]);
    const vector = embeddings.arraySync()[0];

    // Send embedding to Flask backend
    const response = await fetch("/ask", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ vector: vector })
    });

    const data = await response.json();
    botMsg.textContent = data.answer;
  } catch (err) {
    console.error("[ERROR]", err);
    botMsg.textContent = "Oops! Something went wrong.";
  }

  input.value = "";
  chatBox.scrollTop = chatBox.scrollHeight;
}

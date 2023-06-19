<script>
  var text = "Click The Button"
 const getTextFromAudio = () => {
  fetch('http://127.0.0.1:8000/getAudio').then(r => r.json()).then(t => text = t['data'])
 }
</script>

<main>
  <div>
    {text}
  </div>
<button on:click={getTextFromAudio}>Say Something</button>  
</main>


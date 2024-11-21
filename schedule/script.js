document.addEventListener('DOMContentLoaded', () => {
    const messages = document.querySelectorAll('.imessage p');
    let index = 0;

    function showMessage() {
        if (index < messages.length) {
            messages[index].classList.add('show'); 

            const delay = index === messages.length - 1 ? 6000 : 2000; 

            index++;
            setTimeout(showMessage, delay); 
        } else {
            highlightWords();
            document.querySelector('.imessage').classList.add('animation-complete');
        }
    }

    function highlightWords() {
        const wordsToHighlight = [
            { id: 'medication', duration: 2000 },
            { id: 'hydrated', duration: 2000 },
            { id: 'sports', duration: 2000 },
            { id: 'temperature', duration: 2000 }
        ];

        let currentIndex = 0;

        function highlightNextWord() {
            if (currentIndex < wordsToHighlight.length) {
                const word = document.getElementById(wordsToHighlight[currentIndex].id);
                word.classList.add('highlight'); 

                setTimeout(() => {
                    word.classList.remove('highlight'); 
                    highlightNextWord(); 
                }, wordsToHighlight[currentIndex].duration);
                
                currentIndex++;
            }
        }

        highlightNextWord(); 
    }

    if (!localStorage.getItem('animationShown')) {
        showMessage();
        localStorage.setItem('animationShown', 'true');
    } else {
        messages.forEach(message => message.classList.add('show'));
        document.querySelector('.imessage').classList.add('animation-complete');
    }
});
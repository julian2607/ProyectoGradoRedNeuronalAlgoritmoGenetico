// Alertas
const alertPlaceholder = document.getElementById('liveAlertPlaceholder')

const appendAlert = (message, type) => {
const wrapper = document.createElement('div')
wrapper.innerHTML = [
    `<div class="alert alert-${type} alert-dismissible" role="alert">`,
    `   <div>${message}</div>`,
    '   <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>',
    '</div>'
].join('')
alertPlaceholder.append(wrapper)
}

// ALERTAS
const alertTrigger = document.getElementById('liveAlertBtn')
if (alertTrigger) {
alertTrigger.addEventListener('click', () => {
    appendAlert('Nice, you triggered this alert message!', 'success')
})
}

// ACTUALIZAR INFO PAGINA
angular.module('myApp', [])
        .controller('myCtrl', function ($scope, $http, $timeout) {
            $scope.control=true;            
        // Actualizar Front
        function actualizarContador() {  
            $scope.control=false;      
            fetch('/ActualizarInformacionResultados')
                .then(response => response.json())
                .then(data => {
                    //Datos reducicon dimensionalidad
                    try{
                        var tablaHtml = data.TablaEjecutar;
                        document.getElementById('TablaInfoResultados').innerHTML = tablaHtml; 
                    }
                    catch{}

                    //Datos Entrenamiento
                    try{
                        var tablaHtml1 = data.contador;
                        document.getElementById('TablaInfo').innerHTML = tablaHtml1;                                      
                    }catch{}                    
                    
                });                        
        }
        setInterval(actualizarContador, 1000);
});


//PETICION EJECUTAR MODELO GUARDADO
document.getElementById('EjecutarModeloGuardado').addEventListener('submit', function(event) {
    event.preventDefault(); 
    var loadingScreen = document.getElementById('loadingScreen');
    loadingScreen.classList.remove('hidden');
    var formData = new FormData(this);
    fetch('http://127.0.0.1:5000/EvaluarModelo', {
        method: 'POST',
        body: formData
    })    
    .then(data => {
        loadingScreen.classList.add('hidden');        
        modal.classList.remove('hidden')        
        appendAlert('Proceso realizado con exito', 'success');
        modalMessage.textContent = "Ejecucion de la red realizada con exito.";         
        modal.style.display = 'flex';
        console.log(data);
    })
    .catch(error => {
        loadingScreen.classList.add('hidden');
        appendAlert('Ha ocurrido un error ejecutando el modelo', 'error');
        console.error('Error:', error);        
    });
});


//Redirigir a nueva magian con botor
document.getElementById("Iniciar").addEventListener("click", function() {
    // Redirigir a otra página
    window.location.href = "http://127.0.0.1:5000/PaginaIniciar";
    loader.classList.add('hidden');
});

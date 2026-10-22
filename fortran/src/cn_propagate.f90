!> Command-line driver for the Fortran Crank-Nicolson solver.
!>
!> Usage:   ./cn_propagate [config.nml]
!>
!> Reads the namelist &config (all entries optional, defaults below), runs the
!> propagation and writes a CSV file with columns
!>     t, norm, x_mean, overlap_abs
!> where overlap_abs = |<psi(0)|psi(t)>|. A one-line summary is printed on
!> standard output.
program cn_propagate
   use tdse_cn
   implicit none

   ! ---------------- parameters (namelist &config) ----------------
   character(len=16) :: potential = 'softcoulomb'   ! 'softcoulomb' | 'harmonic'
   character(len=16) :: initial = 'ground'          ! 'ground' | 'gaussian'
   character(len=256) :: output = 'fortran_output.csv'
   real(dp) :: x_min = -150.0_dp, x_max = 150.0_dp
   integer :: nx = 3001
   real(dp) :: dt = 0.0_dp                          ! <= 0 => 2 pi / omega / 128
   integer :: n_steps = 2432
   integer :: record_every = 1
   real(dp) :: e0 = 1.0e-2_dp, omega = 1.0_dp       ! laser field
   real(dp) :: n_ramp = 4.0_dp                      ! sin^2 ramp, in optical cycles
   real(dp) :: omega_osc = 1.0_dp                   ! harmonic potential
   real(dp) :: softening = 1.0_dp                   ! soft-Coulomb parameter a
   real(dp) :: x0 = 2.0_dp, sigma = 1.0_dp          ! Gaussian initial state
   real(dp) :: shift = -1.0_dp                      ! inverse-iteration shift
   namelist /config/ potential, initial, output, x_min, x_max, nx, dt, n_steps, &
      record_every, e0, omega, n_ramp, omega_osc, softening, x0, sigma, shift

   ! ---------------- work variables ----------------
   real(dp), allocatable :: x(:), diag(:), diag_t(:), phi(:)
   complex(dp), allocatable :: psi(:), psi_init(:)
   real(dp) :: dx, off, energy, t_mid, t_ramp, field, nrm, xm, ov, max_norm_err
   integer :: k, unit_cfg, unit_out, ios, iterations, i
   integer(8) :: clock_start, clock_end, clock_rate
   logical :: converged
   character(len=256) :: cfg_file

   ! ---------------- read configuration ----------------
   if (command_argument_count() >= 1) then
      call get_command_argument(1, cfg_file)
      open (newunit=unit_cfg, file=trim(cfg_file), status='old', action='read', iostat=ios)
      if (ios /= 0) error stop 'cannot open configuration file'
      read (unit_cfg, nml=config, iostat=ios)
      if (ios /= 0) error stop 'invalid &config namelist'
      close (unit_cfg)
   end if
   if (nx < 3) error stop 'nx must be >= 3'
   if (x_max <= x_min) error stop 'x_max must be > x_min'
   if (record_every < 1) error stop 'record_every must be >= 1'
   if (dt <= 0.0_dp) dt = 2.0_dp*pi/omega/128.0_dp
   t_ramp = n_ramp*2.0_dp*pi/omega

   ! ---------------- grid and field-free Hamiltonian ----------------
   allocate (x(nx), diag(nx), diag_t(nx), phi(nx), psi(nx), psi_init(nx))
   dx = (x_max - x_min)/real(nx - 1, dp)
   x = [(x_min + real(i - 1, dp)*dx, i=1, nx)]
   select case (trim(potential))
   case ('softcoulomb')
      diag = 1.0_dp/dx**2 - 1.0_dp/sqrt(x**2 + softening**2)
   case ('harmonic')
      diag = 1.0_dp/dx**2 + 0.5_dp*omega_osc**2*x**2
   case default
      error stop 'unknown potential (use softcoulomb or harmonic)'
   end select
   off = -0.5_dp/dx**2

   ! ---------------- initial state ----------------
   energy = 0.0_dp
   select case (trim(initial))
   case ('ground')
      phi = 1.0_dp
      call inverse_iteration(diag, off, shift, phi, energy, iterations, converged)
      if (.not. converged) error stop 'inverse iteration did not converge'
      psi = cmplx(phi/sqrt(dx), 0.0_dp, kind=dp)
   case ('gaussian')
      psi = cmplx(exp(-(x - x0)**2/(2.0_dp*sigma**2)), 0.0_dp, kind=dp)
      psi = psi/sqrt(sum(abs(psi)**2)*dx)
   case default
      error stop 'unknown initial state (use ground or gaussian)'
   end select
   psi_init = psi

   ! ---------------- propagation ----------------
   open (newunit=unit_out, file=trim(output), status='replace', action='write', iostat=ios)
   if (ios /= 0) error stop 'cannot open output file'
   write (unit_out, '(a)') 't,norm,x_mean,overlap_abs'
   call write_row(0.0_dp)

   max_norm_err = 0.0_dp
   call system_clock(clock_start, clock_rate)
   do k = 1, n_steps
      t_mid = (real(k - 1, dp) + 0.5_dp)*dt
      field = e0*sin2_ramp(t_mid, t_ramp)*sin(omega*t_mid)
      diag_t = diag + field*x
      call cn_step(diag_t, off, dt, psi)
      if (mod(k, record_every) == 0 .or. k == n_steps) call write_row(real(k, dp)*dt)
   end do
   call system_clock(clock_end)
   close (unit_out)

   if (trim(initial) == 'ground') write (*, '(a,es23.15e3,a,i0,a)') &
      'E_ground=', energy, '  (', iterations, ' inverse iterations)'
   write (*, '(a,i0,a,es10.3,a,f10.4,a)') &
      'steps=', n_steps, '  max|norm-1|=', max_norm_err, &
      '  propagation time=', real(clock_end - clock_start, dp)/real(clock_rate, dp), ' s'

contains

   subroutine write_row(t)
      real(dp), intent(in) :: t
      nrm = sum(abs(psi)**2)*dx
      xm = sum(x*abs(psi)**2)*dx
      ov = abs(sum(conjg(psi_init)*psi)*dx)
      max_norm_err = max(max_norm_err, abs(nrm - 1.0_dp))
      write (unit_out, '(es24.16e3,3(",",es24.16e3))') t, nrm, xm, ov
   end subroutine write_row

end program cn_propagate

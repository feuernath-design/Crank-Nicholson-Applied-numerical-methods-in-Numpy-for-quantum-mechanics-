!> Crank-Nicolson solver for the 1D time-dependent Schrodinger equation.
!>
!> Fortran 2008 port of the Python package `tdse1d` (atomic units).
!>   H(t) = -1/2 d^2/dx^2 + V(x) + E(t) x      (3-point finite differences)
!>   (1 + i dt/2 H(t+dt/2)) psi(t+dt) = (1 - i dt/2 H(t+dt/2)) psi(t)
!> The tridiagonal systems are solved with the Thomas algorithm, O(N) per step.
module tdse_cn
   use, intrinsic :: iso_fortran_env, only: dp => real64
   implicit none
   private

   public :: dp, pi
   public :: thomas_real, thomas_complex
   public :: apply_hamiltonian
   public :: inverse_iteration
   public :: cn_step
   public :: sin2_ramp

   real(dp), parameter :: pi = 4.0_dp*atan(1.0_dp)

contains

   !> Solve the real tridiagonal system with diagonal `d` and constant
   !> off-diagonal `off` (Thomas algorithm, no pivoting).
   pure subroutine thomas_real(d, off, rhs, x)
      real(dp), intent(in) :: d(:), off, rhs(:)
      real(dp), intent(out) :: x(:)
      real(dp) :: cprime(size(d)), rprime(size(d)), denom
      integer :: i, n

      n = size(d)
      cprime(1) = off/d(1)
      rprime(1) = rhs(1)/d(1)
      do i = 2, n
         denom = d(i) - off*cprime(i - 1)
         cprime(i) = off/denom
         rprime(i) = (rhs(i) - off*rprime(i - 1))/denom
      end do
      x(n) = rprime(n)
      do i = n - 1, 1, -1
         x(i) = rprime(i) - cprime(i)*x(i + 1)
      end do
   end subroutine thomas_real

   !> Complex version of `thomas_real`.
   pure subroutine thomas_complex(d, off, rhs, x)
      complex(dp), intent(in) :: d(:), off, rhs(:)
      complex(dp), intent(out) :: x(:)
      complex(dp) :: cprime(size(d)), rprime(size(d)), denom
      integer :: i, n

      n = size(d)
      cprime(1) = off/d(1)
      rprime(1) = rhs(1)/d(1)
      do i = 2, n
         denom = d(i) - off*cprime(i - 1)
         cprime(i) = off/denom
         rprime(i) = (rhs(i) - off*rprime(i - 1))/denom
      end do
      x(n) = rprime(n)
      do i = n - 1, 1, -1
         x(i) = rprime(i) - cprime(i)*x(i + 1)
      end do
   end subroutine thomas_complex

   !> y = H psi for a real vector (tridiagonal H, constant off-diagonal).
   pure subroutine apply_hamiltonian(diag, off, psi, y)
      real(dp), intent(in) :: diag(:), off, psi(:)
      real(dp), intent(out) :: y(:)
      integer :: n

      n = size(diag)
      y = diag*psi
      y(1:n - 1) = y(1:n - 1) + off*psi(2:n)
      y(2:n) = y(2:n) + off*psi(1:n - 1)
   end subroutine apply_hamiltonian

   !> Shifted inverse iteration with accelerated shift update
   !> (same algorithm as tdse1d.eigen.inverse_iteration).
   !> On exit `phi` is normalised to sum(phi**2) = 1 (Euclidean norm).
   subroutine inverse_iteration(diag, off, shift, phi, energy, iterations, converged, tol, max_iter)
      real(dp), intent(in) :: diag(:), off, shift
      real(dp), intent(inout) :: phi(:)       !< in: starting vector
      real(dp), intent(out) :: energy
      integer, intent(out) :: iterations
      logical, intent(out) :: converged
      real(dp), intent(in), optional :: tol
      integer, intent(in), optional :: max_iter

      integer, parameter :: fixed_shift_iterations = 3
      real(dp) :: psi(size(diag)), hphi(size(diag)), eps, overlap, residual, tol_
      integer :: it, max_iter_

      tol_ = 1.0e-10_dp
      if (present(tol)) tol_ = tol
      max_iter_ = 500
      if (present(max_iter)) max_iter_ = max_iter

      eps = shift
      phi = phi/norm2(phi)
      converged = .false.
      energy = 0.0_dp
      iterations = 0
      do it = 1, max_iter_
         iterations = it
         call thomas_real(diag - eps, off, phi, psi)
         if (it > fixed_shift_iterations) then
            overlap = dot_product(phi, psi)
            if (abs(overlap) > tiny(1.0_dp) .and. abs(overlap) < huge(1.0_dp)) eps = eps + 1.0_dp/overlap
         end if
         psi = psi/norm2(psi)
         if (dot_product(phi, psi) < 0.0_dp) psi = -psi
         phi = psi

         call apply_hamiltonian(diag, off, phi, hphi)
         energy = dot_product(phi, hphi)
         residual = norm2(hphi - energy*phi)
         if (residual < tol_*max(1.0_dp, abs(energy))) then
            converged = .true.
            exit
         end if
      end do
   end subroutine inverse_iteration

   !> One Crank-Nicolson step. `diag_t` is the main diagonal of H at the
   !> mid-point t + dt/2 (field included); `off` the constant off-diagonal.
   subroutine cn_step(diag_t, off, dt, psi)
      real(dp), intent(in) :: diag_t(:), off, dt
      complex(dp), intent(inout) :: psi(:)

      complex(dp), parameter :: im = (0.0_dp, 1.0_dp)
      complex(dp) :: rhs(size(psi)), lhs_diag(size(psi)), a
      integer :: n

      n = size(psi)
      a = 0.5_dp*im*dt
      ! rhs = (1 - i dt/2 H) psi
      rhs = (1.0_dp - a*diag_t)*psi
      rhs(1:n - 1) = rhs(1:n - 1) - a*off*psi(2:n)
      rhs(2:n) = rhs(2:n) - a*off*psi(1:n - 1)
      ! solve (1 + i dt/2 H) psi_new = rhs
      lhs_diag = 1.0_dp + a*diag_t
      call thomas_complex(lhs_diag, a*off, rhs, psi)
   end subroutine cn_step

   !> sin^2 turn-on over `t_ramp`, then 1 (t_ramp = 0: sudden switch-on).
   pure real(dp) function sin2_ramp(t, t_ramp) result(f)
      real(dp), intent(in) :: t, t_ramp

      if (t_ramp <= 0.0_dp .or. t >= t_ramp) then
         f = 1.0_dp
      else
         f = sin(pi*max(t, 0.0_dp)/(2.0_dp*t_ramp))**2
      end if
   end function sin2_ramp

end module tdse_cn
